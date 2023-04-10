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
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup,AutoModelForSequenceClassification
from accelerate import Accelerator
import glob
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(name,tokenizer,maxlen):
    subtitle = json.load(open(name[0]+'/subtitle.json',encoding='utf-8'))
    with open('{}/{}'.format(name[0],name[1]), encoding='utf-8') as f:
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
            if len(tokenizer(''.join(n['text'] for n in i['video_sub_title'])).input_ids) > maxlen:
                not_load_example+=1
                continue
            if i['video_id'] in video_id[:int(len(video_id) * 0.1)]:
                valid.append(i)
            else:
                train.append(i)
        except:
            not_load_example += 1

    print('Not Load Example: ',not_load_example)
    return train,valid

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

def pad_video_seq(sequences, max_length=1024):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[1]
        sequence_length.append(seq.shape[1])
        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_length], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length

def get_args(args):
    l = []
    for k in list(vars(args).keys()):
        l.append(('%s: %s' % (k, vars(args)[k])))
    return l


def log(text,path):
    with open(path+'/log.txt','a',encoding='utf-8') as f:
        f.write('-----------------{}-----------------'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        for i in text:
            f.write(i)
            print(i)
            f.write('\n')
        f.write('\n')

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


def collate_fn_video(data):
    input_ids, attention_mask, token_type_ids,start_labels,end_labels,video_features,audios,h_labels,start_times,end_times = [], [], [],[],[],[],[],[],[],[]
    for x in data:
        input_id,attention,token_type = [],[],[]
        sub = x['video_sub_title']
        min_start = 10000
        min_end = 10000
        start_text = x['video_sub_title'][0]['text']
        end_text = x['video_sub_title'][-1]['text']
        for s in range(len(sub)):
            if abs(sub[s]['start']-x['answer_start_second']) < min_start:
                start_text = sub[s]['text']
                start_id = s
                min_start = abs(sub[s]['start']-x['answer_start_second'])
            if abs(sub[s]['start']+sub[s]['duration']-x['answer_end_second']) <= min_end:
                end_text = sub[s]['text']
                end_id = s
                min_end = abs(sub[s]['start']+sub[s]['duration']-x['answer_end_second'])

        text = x['question']
        text = tokenizer(text)
        input_id.extend(text.input_ids)
        token_type.extend([0]*len(text.input_ids))

        for s in range(len(sub)):
            if s == start_id:
                start_label = len(input_id)+1
            input_id.extend(tokenizer(sub[s]['text']).input_ids[1:])
            if s == end_id:
                end_label = len(input_id)-1
        vi = visual[x['video_id']]
        video_features.append(vi)

        h_label = np.zeros(shape=[1024], dtype=np.int32)
        st, et = x['answer_start_second'],x['answer_end_second']
        start_times.append(st)
        end_times.append(et)
        cur_max_len = vi.shape[0]
        extend_len = round(args.highlight_hyperparameter * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_label[st_:(et_ + 1)] = 1
        else:
            h_label[st:(et + 1)] = 1
        h_labels.append(h_label)

        token_type.extend([1] * (len(input_id)-len(token_type)))
        attention = [1] * len(input_id)
        input_ids.append(input_id)
        attention_mask.append(attention)
        token_type_ids.append(token_type)
        start_labels.append(start_label)
        end_labels.append(end_label)

    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats_mask = torch.stack([torch.cat([torch.ones([vi.shape[0]]),torch.zeros([1024-vi.shape[0]])])])
    h_labels=torch.tensor(h_labels)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    vfeats = torch.tensor(vfeats)
    start_labels = torch.tensor(start_labels)
    end_labels = torch.tensor(end_labels)
    start_times = torch.tensor(start_times)
    end_times = torch.tensor(end_times)
    return input_ids, attention_mask, token_type_ids,vfeats,start_labels,end_labels,h_labels,vfeats_mask,start_times,end_times,x