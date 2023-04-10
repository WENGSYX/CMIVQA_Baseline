import json
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from sklearn.model_selection import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup, \
    AutoModelForSequenceClassification
# from transformers.deepspeed import HfDeepSpeedConfig
from torch.autograd import Variable
from accelerate import Accelerator
import glob

accelerator = Accelerator()
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
parser.add_argument("--deepspeed_config", default=-1)
args = parser.parse_args()

CFG = {  # 训练的参数配置
    'seed': 2021,
    'model': r'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese',  # 预训练模型
    'max_len': 200,  # 文本截断的最大长度
    'epochs': 16,
    'train_bs': 1,  # batch_size，可根据自己的显存调整
    'valid_bs': 1,
    'lr': 8e-6,  # 学习率
    'num_workers': 0,
    'accum_iter': 1,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-6,  # 权重衰减，防止过拟合
    'device': 0,
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG['seed'])  # 固定随机种子

torch.cuda.set_device(CFG['device'])
device = accelerator.device



from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(CFG['model'])

def visual_feature_sampling(visual_feature, max_num_clips):
    num_clips = visual_feature.shape[0]
    if num_clips <= max_num_clips:
        return visual_feature
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_visual_feature = []
    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
        else:
            new_visual_feature.append(visual_feature[s_idx])
    new_visual_feature = np.asarray(new_visual_feature)
    return new_visual_feature


def load_video_features(root, max_position_length):
    video_features = dict()
    filenames = glob.glob(os.path.join(root, "*.npy"))
    for filename in tqdm(filenames, total=len(filenames), desc="loading video features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = np.load(filename)
        if max_position_length is None:
            video_features[video_id] = feature
        else:
            new_feature = visual_feature_sampling(feature, max_num_clips=max_position_length)
            video_features[video_id] = new_feature
    return video_features


visual = load_video_features(os.path.join('data', 'train'), 1024)
subtitle = json.load(open('./data/subtitle.json',encoding='utf-8'))

def get_data(name):
    with open('./data/text/{}'.format(name), encoding='utf-8') as f:
        data = json.load(f)
    video_id = list(set([i['video_id'] for i in data]))
    random.shuffle(video_id)
    train,valid = [],[]
    not_load_example = 0
    for i in data:
        try:
            i['video_sub_title'] = subtitle[i['video_id']]
            if len(i['video_sub_title']) <= 2:
                not_load_example += 1
                continue
            i['start_second'] = 0
            i['end_second'] = 0
            valid.append(i)
        except:
            not_load_example += 1
    print('Not Load Example: ',not_load_example)
    return valid


valid = get_data(r'dataset_testA_for_task_1.json')




class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df[idx]

        return data


def pad_video_seq(sequences, max_length=1024):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_length], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length


def collate_fn(data):
    input_ids, attention_mask, token_type_ids, start_labels, end_labels, video_features, h_labels,start_times,end_times = [], [], [], [], [], [], [], [], []
    for x in data:
        input_id, attention, token_type = [], [], []
        sub = x['video_sub_title']
        min_start = 10000
        min_end = 10000
        start_text = x['video_sub_title'][0]['text']
        end_text = x['video_sub_title'][-1]['text']
        for s in range(len(sub)):
            if abs(sub[s]['start'] - x['start_second']) < min_start:
                start_text = sub[s]['text']
                start_id = s
                min_start = abs(sub[s]['start'] - x['start_second'])
            if abs(sub[s]['start'] + sub[s]['duration'] - x['end_second']) <= min_end:
                end_text = sub[s]['text']
                end_id = s
                min_end = abs(sub[s]['start'] + sub[s]['duration'] - x['end_second'])

        text = x['question']
        try:
            vi = visual[x['video_id']]
        except:
            vi = np.zeros((2,1024),dtype=np.float32)
        video_features.append(vi)

        h_label = np.zeros(shape=[1024], dtype=np.int32)
        st, et = x['start_second'], x['end_second']
        start_times.append(st)
        end_times.append(et)
        cur_max_len = vi.shape[0]
        extend_len = round(0.25 * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_label[st_:(et_ + 1)] = 1
        else:
            h_label[st:(et + 1)] = 1

        text = tokenizer(text)
        input_id.extend(text.input_ids)
        token_type.extend([0] * len(text.input_ids))

        for s in range(len(sub)):
            if s == start_id:
                start_label = len(input_id) + 1
            input_id.extend(tokenizer(sub[s]['text']).input_ids[1:])
            if s == end_id:
                end_label = len(input_id) - 1

        token_type.extend([1] * (len(input_id) - len(token_type)))
        attention = [1] * len(input_id)
        h_labels.append(h_label)
        input_ids.append(input_id)
        attention_mask.append(attention)
        token_type_ids.append(token_type)
        start_labels.append(start_label)
        end_labels.append(end_label)

    vfeats, vfeat_lens = pad_video_seq(video_features)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    vfeats = torch.tensor(vfeats)
    start_labels = torch.tensor(start_labels)
    end_labels = torch.tensor(end_labels)
    h_labels = torch.tensor(h_labels)
    start_times = torch.tensor(start_times)
    end_times = torch.tensor(end_times)
    return input_ids, attention_mask, token_type_ids, vfeats, start_labels, end_labels, h_labels, start_times, end_times, x


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import time
import torch.nn as nn



def test_model(model, val_loader, epoch):  # 验证
    model.eval()

    losses = AverageMeter()
    pred_start = []
    pred_end = []
    token_id = []
    tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
    for step, (
    input_ids, attention_mask, token_type_ids, vfeats, start_labels, end_labels, h_labels, start_times, end_times,
    x) in enumerate(tk):
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           video_features=vfeats, start_positions=start_labels, end_positions=end_labels,
                           start_times=start_times, end_times=end_times, data=x,tokenizer=tokenizer)
            loss = output.loss

        losses.update(loss.item(), input_ids.size(0))
        pred_start.extend(output.start_logits)
        pred_end.extend(output.end_logits)
        token_id.extend(input_ids)

    return losses.avg, pred_start, pred_end, token_id


from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr


def score(submission_file: str, reference_file: str, subtask: str) -> float:
    """Assign an overall score to submitted predictions.

    :param submission_file: str path to submission file with predicted ratings
    :param reference_file: str path to file with gold ratings
    :param subtask: str indicating if the predictions are for the ranking or the classification task
    options: 'ranking' or 'classification'
    :return: float score
    """
    predictions = []
    target = []

    submission = pd.read_csv(
        submission_file, sep="\t", header=None, names=["Id", "Label"]
    )

    reference = pd.read_csv(
        reference_file, sep="\t", header=None, names=["Id", "Label"]
    )
    # the reference file must have the same format as the submission file, so we use the same format checker

    if submission.size != reference.size:
        raise ValueError(
            "Submission does not contain the same number of rows as reference file."
        )

    for _, row in submission.iterrows():
        reference_indices = list(reference["Id"][reference["Id"] == row["Id"]].index)

        if not reference_indices:
            raise ValueError(
                f"Identifier {row['Id']} does not appear in reference file."
            )
        elif len(reference_indices) > 1:
            raise ValueError(
                f"Identifier {row['Id']} appears several times in reference file."
            )
        else:
            reference_index = reference_indices[0]

            if subtask == "ranking":
                target.append(float(reference["Label"][reference_index]))
                predictions.append(float(row["Label"]))
            elif subtask == "classification":
                target.append(reference["Label"][reference_index])
                predictions.append(row["Label"])
            else:
                raise ValueError(
                    f"Evaluation mode {subtask} not available: select ranking or classification"
                )

    if subtask == "ranking":
        score = spearmans_rank_correlation(
            gold_ratings=target, predicted_ratings=predictions
        )


    elif subtask == "classification":
        prediction_ints = convert_class_names_to_int(predictions)
        target_ints = convert_class_names_to_int(target)

        score = accuracy_score(y_true=target_ints, y_pred=prediction_ints)


    else:
        raise ValueError(
            f"Evaluation mode {subtask} not available: select ranking or classification"
        )

    return score


def convert_class_names_to_int(labels: List[str]) -> List[int]:
    """Convert class names to integer label indices.

    :param labels:
    :return:
    """
    class_names = ["IMPLAUSIBLE", "NEUTRAL", "PLAUSIBLE"]
    label_indices = []

    for label in labels:
        try:
            label_index = class_names.index(label)
        except ValueError:
            raise ValueError(f"Label {label} is not in label set {class_names}.")
        else:
            label_indices.append(label_index)

    return label_indices


def spearmans_rank_correlation(
        gold_ratings: List[float], predicted_ratings: List[float]
) -> float:
    """Score submission for the ranking task with Spearman's rank correlation.

    :param gold_ratings: list of float gold ratings
    :param predicted_ratings: list of float predicted ratings
    :return: float Spearman's rank correlation coefficient
    """
    if len(gold_ratings) == 1 and len(predicted_ratings) == 1:
        raise ValueError("Cannot compute rank correlation on only one prediction.")

    return spearmanr(a=gold_ratings, b=predicted_ratings)[0]


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name1='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name1='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, i, t, f):
        if self.logits:
            # targets_list=t.tolist()
            targets = t
            inputs = i
            # for tar in range(len(targets_list)):
            # for p in range(f[targets_list[tar]]):
            # targets.append(targets_list[tar])
            # inputs.append(i[tar].tolist())
            # inputs = torch.tensor(inputs,device=device)
            # targets = torch.tensor(targets)
            targets = torch.eye(35)[targets.reshape(-1)].to(device)
            topic_weight = torch.ones_like(targets) + targets * 6
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False, weight=topic_weight)
        else:
            targets = torch.eye(35)[targets.reshape(-1)].to(device)
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


import torch.nn as nn


class CB_loss(nn.Module):
    def __init__(self, beta, gamma, epsilon=0.1):
        super(CB_loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits, labels, loss_type='focal'):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        # self.epsilon = 0.1 #labelsmooth
        beta = self.beta
        gamma = self.gamma

        no_of_classes = logits.shape[1]
        samples_per_cls = torch.Tensor([sum(labels == i) for i in range(logits.shape[1])])
        if torch.cuda.is_available():
            samples_per_cls = samples_per_cls.cuda()

        effective_num = 1.0 - torch.pow(beta, samples_per_cls)
        weights = (1.0 - beta) / ((effective_num) + 1e-8)
        # print(weights)
        weights = weights / torch.sum(weights) * no_of_classes
        labels = labels.reshape(-1, 1)

        labels_one_hot = torch.zeros(len(labels), no_of_classes).scatter_(1, labels.cpu(), 1)

        weights = torch.tensor(weights).float()
        if torch.cuda.is_available():
            weights = weights.cuda()
            labels_one_hot = torch.zeros(len(labels), no_of_classes).cuda().scatter_(1, labels, 1).cuda()

        labels_one_hot = (1 - self.epsilon) * labels_one_hot + self.epsilon / no_of_classes
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, no_of_classes)

        if loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, pos_weight=weights)
        elif loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss


def log(text, path):
    with open(path + '/log.txt', 'a', encoding='utf-8') as f:
        f.write('-----------------{}-----------------'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        for i in text:
            f.write(i)
            print(i)
            f.write('\n')
        f.write('\n')


import os


def log_start(log_name):
    if log_name == '':
        log_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    try:
        os.mkdir('paperlog/' + log_name)
    except:
        log_name = log_name + '_'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        os.mkdir('paperlog/' + log_name)

    with open('paperlog/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__), 'r', encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)
    with open('paperlog/' + log_name + '.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__), 'r', encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    path = 'paperlog/' + log_name
    with open(path + '/log.txt', 'a', encoding='utf-8') as f:
        f.write(log_name)
        f.write('\n')
    return path


def macro_f1(pred, gold):
    intersection = 0
    GOLD = 0
    PRED = 0
    intersection1 = 0
    GOLD1 = 0
    PRED1 = 0
    F1s = []
    f = []
    for l in range(3):
        l_p = []
        l_g = []
        for i in range(len(pred)):
            p = pred[i]
            g = gold[i]

            if g == l:
                l_g.append(i)
            if p == l:
                l_p.append(i)

        l_g = set(l_g)
        l_p = set(l_p)

        TP = len(l_g & l_p)
        FP = len(l_p) - TP
        FN = len(l_g) - TP
        precision = TP / (TP + FP + 0.0000000001)
        recall = TP / (TP + FN + 0.0000000001)
        F1 = (2 * precision * recall) / (precision + recall + 0.0000000001)
        F1s.append(F1)
    return F1s


def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)



valid_set = MyDataset(valid)


valid_loader = DataLoader(valid_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                          num_workers=CFG['num_workers'])

from model import MutualSLModel

model = MutualSLModel.from_pretrained(CFG['model'])
model.load_state_dict(torch.load('/home/hnu1/.ss/NLPCC/task1/paperlog/MutualSL_large/4_62.5_44.26_model/pytorch_model.bin'))
model = model.to(device)

val_loader = accelerator.prepare(valid_loader)
# model,optimizer,_,scheduler = deepspeed.initialize(model=model,config_params=ds_config,optimizer=optimizer,lr_scheduler=scheduler)

log_name = 'MutualSL_large'

val_loss, pred_start, pred_end, token_id = test_model(model, val_loader, 0)
outputs = []
for i in range(len(token_id)):
    start = pred_start[i].argmax().item()
    end = pred_end[i].argmax().item()
    token = token_id[i].tolist()
    tokens = []
    ts = []

    for t_num in range(len(token)):
        ts.append(token[t_num])
        if t_num == start:
            start_time = valid[i]['video_sub_title'][len(tokens) - 1]['start']
        if t_num == end:
            end_time = valid[i]['video_sub_title'][len(tokens) - 1]['start'] + \
                       valid[i]['video_sub_title'][len(tokens) - 1]['duration']
        if token[t_num] == 2:
            tokens.append(ts)
            ts = []
    try:
        if start_time >= end_time:
            end_time = valid[i]['video_sub_title'][-1]['start'] + valid[i]['video_sub_title'][-1]['duration']
    except:
        start_time = valid[i]['video_sub_title'][0]['start']
        end_time = valid[i]['video_sub_title'][-1]['start'] + valid[i]['video_sub_title'][-1]['duration']
    item = valid[i]
    item['start_second'] = start_time
    item['end_second'] = end_time
    outputs.append(item)

with open('output_task1.json','w',encoding='utf-8') as f:
    json.dump(outputs,f,indent=4,ensure_ascii=False)

