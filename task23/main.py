import json
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm,trange
import random
import os
import time
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup,AutoModelForSequenceClassification
from accelerate import Accelerator
import glob
from utils import *

def get_iou(start,end,token,step):
    token = token.tolist()
    tokens = []
    ts = []

    t_num = -1
    for num in range(len(token)):
        ts.append(token[num])
        if t_num == start:
            start_time = valid[step]['video_sub_title'][len(tokens) - 1]['start']
        if t_num == end:
            end_time = valid[step]['video_sub_title'][len(tokens) - 1]['start'] + \
                       valid[step]['video_sub_title'][len(tokens) - 1]['duration']
        if token[num] == 2:
            tokens.append(ts)
            ts = []
            t_num += 1
    if start_time >= end_time:
        end_time = valid[step]['video_sub_title'][-1]['start'] + valid[step]['video_sub_title'][-1]['duration']

    return calculate_iou(i0=[start_time, end_time], i1=[valid[step]["start_second"], valid[step]["end_second"]])


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        data = self.df[idx]

        return [data,self.df]
def collate_fn(data):
    p_input_ids, p_attention_mask, token_types,q_input_ids,q_attention_mask,start_labels,end_labels,target,video_features= [],[],[],[],[],[],[],[],[]
    for data_x in data:
        x = data_x[0]
        input_id,attention= [tokenizer.cls_token_id],[]
        sub = x['video_sub_title']
        min_start = 10000
        min_end = 10000
        start_text = x['video_sub_title'][0]['text']
        end_text = x['video_sub_title'][-1]['text']
        for s in range(len(sub)):
            if abs(sub[s]['start']-x['start_second']) < min_start:
                start_text = sub[s]['text']
                start_id = s
                min_start = abs(sub[s]['start']-x['start_second'])
            if abs(sub[s]['start']+sub[s]['duration']-x['end_second']) <= min_end:
                end_text = sub[s]['text']
                end_id = s
                min_end = abs(sub[s]['start']+sub[s]['duration']-x['end_second'])
        try:
            vi = visual[x['video_id']][0]
        except:
            vi = np.zeros(768,dtype=np.float32)
        video_features.append(vi)
        text = x['question']
        text = tokenizer(text)
        input_id.extend(text.input_ids)
        token_type = [0]*(len(input_id)-1)+[1]
        ious = []
        for s in range(len(sub)):
            iou_item = []
            if s == start_id:
                start_label = sum(token_type)-1
            ids = tokenizer(sub[s]['text']).input_ids[1:]
            token_type.extend([0]*(len(ids)-1))
            token_type.extend([1])
            input_id.extend(ids)
            if s == end_id:
                end_label = sum(token_type)-1

            for s2 in range(len(sub)):
                if s2>=s:
                    iou = calculate_iou(i0=[sub[s]['start'],sub[s2]['start']+sub[s2]['duration']],
                                        i1=[x["start_second"], x["end_second"]])
                    iou_item.append(iou)
                else:
                    iou_item.append(0)
            ious.append(iou_item)
        attention = [1] * len(input_id)
        input_id1 = input_id
        attention1 = attention

        start_label1 = start_label
        end_label1 = end_label
        token_type1 = token_type

        start_labels.append(start_label)
        end_labels.append(end_label)
    token_type = []
    for data_x in data:
        x = data_x[0]
        try:
            vi = visual[x['video_id']][0]
        except:
            vi = np.zeros(768,dtype=np.float32)
        video_features.append(vi)
        input_id,attention= [tokenizer.cls_token_id],[]
        sub = x['video_sub_title']
        min_start = 10000
        min_end = 10000
        start_text = x['video_sub_title'][0]['text']
        end_text = x['video_sub_title'][-1]['text']
        for s in range(len(sub)):
            if abs(sub[s]['start']-x['start_second']) < min_start:
                start_text = sub[s]['text']
                start_id = s
                min_start = abs(sub[s]['start']-x['start_second'])
            if abs(sub[s]['start']+sub[s]['duration']-x['end_second']) <= min_end:
                end_text = sub[s]['text']
                end_id = s
                min_end = abs(sub[s]['start']+sub[s]['duration']-x['end_second'])

        text = questions[random.randint(0,len(questions)-1)]
        text = tokenizer(text)
        input_id.extend(text.input_ids)
        token_type = [0]*(len(input_id)-1)+[1]
        for s in range(len(sub)):
            iou_item = []
            if s == start_id:
                start_label = sum(token_type)-1
            ids = tokenizer(sub[s]['text']).input_ids[1:]
            token_type.extend([0]*(len(ids)-1))
            token_type.extend([1])
            input_id.extend(ids)
            if s == end_id:
                end_label = sum(token_type)-1

            for s2 in range(len(sub)):
                if s2>=s:
                    iou = calculate_iou(i0=[sub[s]['start'],sub[s2]['start']+sub[s2]['duration']],
                                        i1=[x["start_second"], x["end_second"]])
                    iou_item.append(iou)
                else:
                    iou_item.append(0)
        maxlen = max(len(input_id1),len(input_id))
        attention = [1] * len(input_id)
        input_id2 = input_id
        attention2 = attention
        token_type2 = token_type

        p_input_ids.append(input_id1+[tokenizer.pad_token_id] * (maxlen-len(input_id1)))
        p_attention_mask.append(attention1+ [0] * (maxlen-len(input_id1)))
        token_types.append(token_type1+[0] * (maxlen-len(input_id1)))

        p_input_ids.append(input_id2 + [tokenizer.pad_token_id] * (maxlen - len(input_id2)))
        p_attention_mask.append(attention2 + [0] * (maxlen - len(input_id2)))
        token_types.append(token_type2 + [0] * (maxlen - len(input_id2)))


    vfeats = video_features
    vfeats = torch.tensor(vfeats)
    vfeats_mask = torch.tensor([[1]*vfl+[0]*(768-vfl) for vfl in [786]*vfeats.size(0)])
    p_input_ids = torch.tensor(p_input_ids)
    p_attention_mask = torch.tensor(p_attention_mask)
    token_types = torch.tensor(token_types)
    start_labels = torch.tensor(start_labels[0])
    end_labels = torch.tensor(end_labels[0])
    ious = torch.tensor(ious)
    return p_input_ids, p_attention_mask, token_types,ious,vfeats,vfeats_mask
def collate_fn_test(data):
    p_input_ids, p_attention_mask, token_types,start_labels,end_labels,target,video_features= [],[],[],[],[],[],[]
    for data_x in data:
        nums = 0
        video_ids = []
        x = data_x[0]
        for video in data_x[1]:
            if video['video_id'] not in video_ids:
                video_ids.append(video['video_id'])
                try:
                    vi = visual[x['video_id']][0]
                except:
                    vi = np.zeros(768, dtype=np.float32)
                video_features.append(vi)
                if video['video_id'] == x['video_id']:
                    target.append(nums)
                    input_id,attention= [tokenizer.cls_token_id],[]
                    sub = x['video_sub_title']
                    min_start = 10000
                    min_end = 10000
                    start_text = x['video_sub_title'][0]['text']
                    end_text = x['video_sub_title'][-1]['text']
                    for s in range(len(sub)):
                        if abs(sub[s]['start']-x['start_second']) < min_start:
                            start_text = sub[s]['text']
                            start_id = s
                            min_start = abs(sub[s]['start']-x['start_second'])
                        if abs(sub[s]['start']+sub[s]['duration']-x['end_second']) <= min_end:
                            end_text = sub[s]['text']
                            end_id = s
                            min_end = abs(sub[s]['start']+sub[s]['duration']-x['end_second'])

                    text = x['question']
                    text = tokenizer(text)
                    input_id.extend(text.input_ids)
                    token_type = [0]*(len(input_id)-1)+[1]
                    ious = []
                    for s in range(len(sub)):
                        if s == start_id:
                            start_label = sum(token_type)-1
                        ids = tokenizer(sub[s]['text']).input_ids[1:]
                        token_type.extend([0]*(len(ids)-1))
                        token_type.extend([1])
                        input_id.extend(ids)
                        if s == end_id:
                            end_label = sum(token_type)-1

                        for s2 in range(len(sub)):
                            if s2>=s:
                                iou = calculate_iou(i0=[sub[s]['start'],sub[s2]['start']+sub[s2]['duration']],
                                                    i1=[x["start_second"], x["end_second"]])
                                ious.append(iou)
                    attention = [1] * len(input_id)
                    p_input_ids.append(input_id)
                    p_attention_mask.append(attention)

                    start_labels.append(start_label)
                    end_labels.append(end_label)
                    token_types.append(token_type)
                else:
                    input_id, attention = [tokenizer.cls_token_id], []
                    sub = video['video_sub_title']
                    min_start = 10000
                    min_end = 10000

                    text = x['question']
                    text = tokenizer(text)
                    input_id.extend(text.input_ids)
                    token_type = [0] * (len(input_id) - 1) + [1]
                    ious = []
                    for s in range(len(sub)):
                        ids = tokenizer(sub[s]['text']).input_ids[1:]
                        token_type.extend([0] * (len(ids) - 1))
                        token_type.extend([1])
                        input_id.extend(ids)

                    attention = [1] * len(input_id)
                    p_input_ids.append(input_id)
                    p_attention_mask.append(attention)

                    token_types.append(token_type)
                nums+=1

    vfeats = video_features
    vfeats = torch.tensor(vfeats)
    vfeats_mask = torch.tensor([[1]*vfl+[0]*(768-vfl) for vfl in [786]*vfeats.size(0)])
    p_input_ids = [torch.tensor(p_input_id) for p_input_id in p_input_ids]
    p_attention_mask = [torch.tensor(p_attention) for p_attention in p_attention_mask]
    token_types = [torch.tensor(token_type) for token_type in token_types]
    ious = torch.tensor(ious)
    target = torch.tensor(target)
    return p_input_ids, p_attention_mask, token_types,ious,target,vfeats,vfeats_mask


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
def log_start(log_name):
    if log_name == '':
        log_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    try:
        os.mkdir('log/' + log_name)
    except:
        log_name = log_name+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        os.mkdir('log/' + log_name)

    with open('log/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)
    with open('log/' + log_name + '.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    path = 'log/' + log_name
    with open(path+'/log.txt', 'a', encoding='utf-8') as f:
        f.write(log_name)
        f.write('\n')
    return path

def train_model(model,train_loader):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, (input_ids, attention_mask, token_types,ious,vfeats,vfeats_mask) in enumerate(tk):
        if input_ids.size(1)>CFG['max_len']:
            continue
        with autocast():
            output = model(input_ids=input_ids,attention_mask=attention_mask,token_types=token_types,
                          ious=ious,vfeats=vfeats,vfeats_mask=vfeats_mask)


        loss = output['CEloss']
        scaler.scale(loss).backward()

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        losses.update(loss.item())
        tk.set_postfix(loss=losses.avg)
        if step == 0:
            log(['Start Train:','Now epoch:{}'.format(epoch),'Now Loss：{}'.format(str(loss.item())),'all of the step:{}'.format(len(tk))],path)

    log(['Now Loss：{}'.format(str(loss.item())),'Avg Loss：{}'.format(losses.avg),'End this round of training'],path)
    return losses.avg


def test_model(model, val_loader):  # 验证
    model.eval()

    losses = AverageMeter()
    accs_start = AverageMeter()
    accs_end = AverageMeter()
    f1s  = AverageMeter()
    optimizer.zero_grad()
    video_logits = []

    acc, mrr, iou_1s, iou_10s, iou_100s = [],[],[],[],[]
    tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
    for step, (input_ids, attention_mask, token_types,ious,target,vfeats,vfeats_mask) in enumerate(tk):
        logits = []
        ps,pe = [],[]
        ns_dict = {}
        ns = 0
        with torch.no_grad():
            for i in range(len(input_ids)):
                output = model.forward_test(input_ids=input_ids[i].unsqueeze(dim=0),
                                            attention_mask=attention_mask[i].unsqueeze(dim=0),
                                            token_types=token_types[i].unsqueeze(dim=0),vfeats=vfeats[i].unsqueeze(dim=0),vfeats_mask=vfeats_mask[i].unsqueeze(dim=0)
                                            )
                ls = output['logits'].view(-1)
                for ln in range(len(ls)):
                    ns_dict[ns] = [i, int(ln / output['logits'].size(0)), int(ln % output['logits'].size(0))]
                    ns += 1
                logits.extend(ls)
                video_logits.append(output['logits'][output['start'], output['end']])
                ps.append(output['start'])
                pe.append(output['end'])
        a, b = torch.stack(logits).sort(descending=True)
        iou1, iou10, iou100 = [], [], []
        for n in b[:1]:
            if ns_dict[n.item()][0] in target:
                iou1.append(
                    get_iou(ns_dict[n.item()][1], ns_dict[n.item()][2], input_ids[ns_dict[n.item()][0]], step))
                acc.append(1)
            else:
                iou1.append(0)
                acc.append(0)

        for n in b[:10]:
            if ns_dict[n.item()][0] in target:
                iou10.append(
                    get_iou(ns_dict[n.item()][1], ns_dict[n.item()][2], input_ids[ns_dict[n.item()][0]], step))
            else:
                iou10.append(0)

        for n in b[:100]:
            if ns_dict[n.item()][0] in target:
                iou100.append(
                    get_iou(ns_dict[n.item()][1], ns_dict[n.item()][2], input_ids[ns_dict[n.item()][0]], step))
            else:
                iou100.append(0)

        iou_1s.append(max(iou1))
        iou_10s.append(max(iou10))
        iou_100s.append(max(iou100))
        _, b = torch.stack(video_logits).sort(descending=True)
        mrr_n = 1
        for n in b:
            if n.item() in target:
                mrr.append(1 / mrr_n)
                mrr_n += 1

        tk.set_postfix(i1=sum(iou_1s) /len(iou_1s), i10=sum(iou_10s) / len(iou_1s), i100=sum(iou_100s) / len(iou_1s))

    return iou_1s, iou_10s, iou_100s, mrr, acc



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default='base', type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--maxlen", default=750, type=int)
    parser.add_argument("--epochs", default=32, type=int)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--device", default=1, type=float)
    args = parser.parse_args()
    CFG = {
        'seed': args.seed,
        'model': 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese',
        'max_len': args.maxlen,
        'epochs': args.epochs,
        'train_bs': 1,
        'valid_bs': 1,
        'lr': args.lr,
        'num_workers': args.num_workers,
        'accum_iter': args.batchsize,
        'weight_decay': args.weight_decay,
        'device': args.device,
    }

    accelerator = Accelerator()
    seed_everything(CFG['seed'])
    torch.cuda.set_device(CFG['device'])
    device = accelerator.device
    visual = load_video_features(os.path.join('data', 'train'), 768)

    tokenizer = AutoTokenizer.from_pretrained(CFG['model'])
    train,valid = get_data(['./NLPCC_2023_CMIVQA_TRAIN_DEV','CMIVQA_Train_Dev.json'],tokenizer,args.maxlen)
    questions = [x['question'] for x in train]


    train_set = MyDataset(train)
    valid_set = MyDataset(valid)

    train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                              num_workers=CFG['num_workers'])
    valid_loader = DataLoader(valid_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn_test, shuffle=False,
                              num_workers=CFG['num_workers'])

    best_acc = 0

    from model import GlobalSpanModel
    model = GlobalSpanModel(CFG['model'],768)


    model = model.to(device)

    deberta_parameters,video_parameters = [],[]
    for n, p in model.named_parameters():
        if "encoder" in n:
            deberta_parameters.append(p)
        else:
            video_parameters.append(p)
    optimizer_parameters = [
        {
            "params": deberta_parameters,
            "lr": CFG['lr'],
        },
        {
            "params": video_parameters,
            "lr": 1e-4,
        },
    ]
    optimizer = AdamW(optimizer_parameters, weight_decay=CFG['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                                CFG['epochs'] * len(train_loader) // CFG['accum_iter'])

    train_loader,val_loader = accelerator.prepare(train_loader,valid_loader)
    scaler = GradScaler()
    log_name = 'Global-Span'
    path = log_start(log_name)
    log(get_args(args),path)


    for epoch in range(CFG['epochs']):
        loss = train_model(model,train_loader)
        iou_1s, iou_10s, iou_100s, mrr, acc = test_model(model, val_loader)

        r1i3 = calculate_iou_accuracy(iou_1s, threshold=0.3)
        r1i5 = calculate_iou_accuracy(iou_1s, threshold=0.5)
        r1i7 = calculate_iou_accuracy(iou_1s, threshold=0.7)
        mi1 = np.mean(iou_1s) * 100.0

        r10i3 = calculate_iou_accuracy(iou_10s, threshold=0.3)
        r10i5 = calculate_iou_accuracy(iou_10s, threshold=0.5)
        r10i7 = calculate_iou_accuracy(iou_10s, threshold=0.7)
        mi10 = np.mean(iou_10s) * 100.0

        r100i3 = calculate_iou_accuracy(iou_100s, threshold=0.3)
        r100i5 = calculate_iou_accuracy(iou_100s, threshold=0.5)
        r100i7 = calculate_iou_accuracy(iou_100s, threshold=0.7)
        mi100 = np.mean(iou_100s) * 100.0
        # write the scores
        score_str = ["Epoch {}".format(epoch)]
        score_str += ['R@ACC: {:.2f}'.format(sum(acc)/len(acc))]
        score_str += ["Rank@1, IoU=0.3: {:.2f}".format(r1i3)]
        score_str += ["Rank@1, IoU=0.5: {:.2f}".format(r1i5)]
        score_str += ["Rank@1, IoU=0.7: {:.2f}".format(r1i7)]
        score_str += ["Rank@1, mean IoU: {:.2f}".format(mi1)]
        score_str += ["Rank@10, IoU=0.3: {:.2f}".format(r10i3)]
        score_str += ["Rank@10, IoU=0.5: {:.2f}".format(r10i5)]
        score_str += ["Rank@10, IoU=0.7: {:.2f}".format(r10i7)]
        score_str += ["Rank@10, mean IoU: {:.2f}".format(mi10)]
        score_str += ["Rank@100, IoU=0.3: {:.2f}".format(r100i3)]
        score_str += ["Rank@100, IoU=0.5: {:.2f}".format(r100i5)]
        score_str += ["Rank@100, IoU=0.7: {:.2f}".format(r100i7)]
        score_str += ["Rank@100, mean IoU: {:.2f}".format(mi100)]
        score_str += ["MRR: {:.2f}".format(sum(mrr)/len(mrr))]
        log(score_str,path)
        model_name = path+'/{}_{}_{}model'.format(epoch,sum(acc)/len(acc),mi1)
        os.mkdir(model_name)
        torch.save(model.state_dict(),model_name+'/pytorch_model.bin')
        tokenizer.save_pretrained(model_name)
        log(score_str+['SAVE MODEL:{}'.format(model_name)],path)
