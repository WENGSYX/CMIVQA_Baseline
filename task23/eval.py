import json
import numpy as np


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


def get_task1_score(target, pred):
    ious = []
    for index in range(len(target)):
        id = target[index]['id']
        pred_item = ''
        for p in pred:
            if p['id'] == id:
                pred_item = p

        if pred_item == '':
            ious.append(0)
        else:
            iou = calculate_iou(i0=[target[index]['start_second'], target[index]['end_second']],
                                i1=[pred_item["start_second"], pred_item["end_second"]])
            ious.append(iou)

    i3 = calculate_iou_accuracy(ious, threshold=0.3)
    i5 = calculate_iou_accuracy(ious, threshold=0.5)
    i7 = calculate_iou_accuracy(ious, threshold=0.7)
    miou = np.mean(ious) * 100.0

    return i3, i5, i7, miou


def get_task2_score(target, pred):
    r1, r10, r50, mrr = [], [], [], []
    for index in range(len(target)):
        id = target[index]['id']
        pred_item = ''
        for p in pred:
            if p['id'] == id:
                pred_item = p
        if pred_item == '':
            r1.append(0)
            r10.append(0)
            r50.append(0)
        else:
            if target[index]['video_id'] in [n[0] for n in pred_item['output'][:1]]:
                r1.append(1)
            else:
                r1.append(0)

            if target[index]['video_id'] in [n[0] for n in pred_item['output'][:10]]:
                r10.append(1)
            else:
                r10.append(0)

            if target[index]['video_id'] in [n[0] for n in pred_item['output'][:50]]:
                r50.append(1)
            else:
                r50.append(0)

            mrr_n = 1
            mrr_pre = []
            for n in [n[0] for n in pred_item['output'][:50]]:
                if n == target[index]['video_id']:
                    mrr_pre.append(1 / mrr_n)
                mrr_n += 1
            if mrr_pre == []:
                mrr.append(0)
            else:
                mrr.append(max(mrr_pre))
    return sum(r1)/len(r1),sum(r10)/len(r10),sum(r50)/len(r50),sum(mrr)/len(mrr)




def get_task3_score(target, pred):
    r1s, r10s, r50s = [], [], []
    for index in range(len(target)):
        r1, r10, r50 = [], [], []
        id = target[index]['id']
        pred_item = ''
        for p in pred:
            if p['id'] == id:
                pred_item = p
        if pred_item == '':
            r1s.append(0)
            r10s.append(0)
            r50s.append(0)
        else:
            for n in pred_item['output'][:1]:
                if n[0] == target[index]['video_id']:
                    r1.append(calculate_iou(i0=[target[index]['start_second'], target[index]['end_second']],
                                 i1=[n[1],n[2]]))
                else:
                    r1.append(0)

            for n in pred_item['output'][:10]:
                if n[0] == target[index]['video_id']:
                    r10.append(calculate_iou(i0=[target[index]['start_second'], target[index]['end_second']],
                                i1=[n[1],n[2]]))
                else:
                    r10.append(0)

            for n in pred_item['output'][:50]:
                if n[0] == target[index]['video_id']:
                    r50.append(calculate_iou(i0=[target[index]['start_second'], target[index]['end_second']],
                                           i1=[n[1], n[2]]))
                else:
                    r50.append(0)

            r1s.append(max(r1))
            r10s.append(max(r10))
            r50s.append(max(r50))


    return sum(r1s)/len(r1s),sum(r10s)/len(r10s),sum(r50s)/len(r50s)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--task', default=1)
    parser.add_argument('--pred_file', type=str, default='output_task1.json')
    parser.add_argument('--target_file', type=str, default='dataset_testA_for_task_1_answer.json')
    args = parser.parse_args()

    if args.task == 1:
        with open(args.pred_file, 'r', encoding='utf-8') as f:
            pred = json.load(f)
        with open(args.target_file, 'r', encoding='utf-8') as f:
            target = json.load(f)
        i3, i5, i7, miou = get_task1_score(target, pred)
        print('R@1,IoU=0.3: {}\nR@1,IoU=0.5: {}\nR@1,IoU=0.7: {}\nmIoU: {}'.format(i3, i5, i7, miou))

    elif args.task == 2:
        with open(args.pred_file, 'r', encoding='utf-8') as f:
            pred = json.load(f)
        with open(args.target_file, 'r', encoding='utf-8') as f:
            target = json.load(f)

        r1,r10,r50,mrr = get_task2_score(target, pred)
        print('R1: {}\nR10: {}\nR50: {}\nMRR: {}'.format(r1,r10,r50,mrr))

    elif args.task == 3:
        with open(args.pred_file, 'r', encoding='utf-8') as f:
            pred = json.load(f)
        with open(args.target_file, 'r', encoding='utf-8') as f:
            target = json.load(f)

        r1,r10,r50 = get_task3_score(target, pred)
        print('R1: {}\nR10: {}\nR50: {}\nAvg: {}'.format(r1,r10,r50,(r1+r10+r50)/3))
