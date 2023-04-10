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

def get_time(start,end,token,step):
    token = token.tolist()
    tokens = []
    ts = []

    t_num = -1
    for num in range(len(token)):
        ts.append(token[num])
        if t_num == start:
            start_time = subtitle[step][len(tokens) - 1]['start']
        if t_num == end:
            end_time = subtitle[step][len(tokens) - 1]['start'] + \
                       subtitle[step][len(tokens) - 1]['duration']
        if token[num] == 2:
            tokens.append(ts)
            ts = []
            t_num += 1
    if start_time >= end_time:
        end_time = subtitle[step][-1]['start'] + subtitle[step][-1]['duration']

    return [start_time, end_time]


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        data = self.df[idx]

        return data



def collate_fn_test(data):
    p_input_ids, p_attention_mask, token_types,start_labels,end_labels,video_ids,video_features= [],[],[],[],[],[],[]
    for x in data:
        nums = 0
        for video in visual.keys():
            vi = visual[video][0]
            video_features.append(vi)
            input_id, attention = [tokenizer.cls_token_id], []
            if video in subtitle:
                if len(subtitle[video]) <= 2:
                    continue
                sub = subtitle[video]
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

    vfeats = video_features
    vfeats = torch.tensor(vfeats)
    vfeats_mask = torch.tensor([[1]*vfl+[0]*(768-vfl) for vfl in [786]*vfeats.size(0)])
    p_input_ids = [torch.tensor(p_input_id) for p_input_id in p_input_ids]
    p_attention_mask = [torch.tensor(p_attention) for p_attention in p_attention_mask]
    token_types = [torch.tensor(token_type) for token_type in token_types]
    return p_input_ids, p_attention_mask, token_types,vfeats,vfeats_mask


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


def test_model(model, val_loader):  # 验证
    model.eval()
    video_logits = []
    outputs = []
    tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
    for step, (input_ids, attention_mask, token_types,vfeats,vfeats_mask) in enumerate(tk):
        logits = []
        ps,pe = [],[]
        ns_dict = {}
        ops = []
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
        for n in b[:50]:
            ops.append([video_ids[ns_dict[n.item()][0]]] +get_time(ns_dict[n.item()][1], ns_dict[n.item()][2], input_ids[ns_dict[n.item()][0]], video_ids[ns_dict[n.item()][0]]))
        outputs.append({'id':step,'question':valid[step]['question'],'output':ops})

    return outputs



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default='base', type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--maxlen", default=1300, type=int)
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
    visual = load_video_features(os.path.join('NLPCC_2023_CMIVQA_TESTA', 'video_feature'), 768)
    subtitle = json.load(open('./NLPCC_2023_CMIVQA_TESTA/subtitle.json',encoding='utf-8'))

    video_ids = []
    for video in visual.keys():
        if video in subtitle:
            if len(subtitle[video]) <= 2:
                continue
            video_ids.append(video)

    tokenizer = AutoTokenizer.from_pretrained(CFG['model'])
    with open('./NLPCC_2023_CMIVQA_TESTA/dataset_testA_for_track23.json', encoding='utf-8') as f:
        valid = json.load(f)

    valid_set = MyDataset(valid)

    valid_loader = DataLoader(valid_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn_test, shuffle=False,
                              num_workers=CFG['num_workers'])

    from model import GlobalSpanModel
    model = GlobalSpanModel(CFG['model'],768)
    model.load_state_dict(torch.load('/home/hnu1/.ss/NLPCC/task23/log/Global-Span/12_0.6_27.028787167554512model/pytorch_model.bin')) #训练完成的模型


    model = model.to(device)

    val_loader = accelerator.prepare(valid_loader)
    scaler = GradScaler()
    log_name = 'Global-Span'


    output = test_model(model, val_loader)
    with open('output_task23.json','w',encoding='utf-8') as f:
        json.dump(output,f,indent=4,ensure_ascii=False)