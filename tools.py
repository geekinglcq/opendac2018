import numpy as np
import pickle as pkl
import json
from XMeans import XMeans
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# from opendac2018 import clustering
from settings import assignments_train_path, pubs_validate_path, \
    local_output_path, global_output_path, pos_pair_path
# todo:
# from RNN_estimate import get_clusters
# from local import make_input

assignments_train_path = './data/assignment_train.json'
pubs_validate_path = './data/pubs_validate.json'

def label2assign(id, y_pred):
    '''
    传入paper id 及预测簇编号
    返回assignment形式：[[id1, id2, ...], [id1, id2, ...]]
    '''
    d = defaultdict(list)
    for i in range(len(id)):
        d[y_pred[i]].append(id[i])
    return list(d.values())

def assign2label(lst):
    '''
    传入一个作者的assignment.
    lst: [ [id1, id2, ...], [id1, id2, ...]  ]
    返回两个list: [id1, id2, id3, ...]; [lab1, lab2, lab3, ...]. 相同簇的paper id具有相同的编号
    '''
    L = 0
    clusters = [c for c in lst]
    id2lab = defaultdict(list)
    for c in clusters:
        for id in c:
            id2lab[id].append(L)
        L += 1
    return list(id2lab.keys()), list(id2lab.values())

def prec_based_on_rule(ass_path):
    """
    Calculate the precision of a assignment with the pos_pair info.
    """
    assigns = json.load(open(ass_path))
    pos_pair = json.load(open(pos_pair_path))
    prec = {}
    for name  in assigns.keys():
        pairs = pos_pair[name]
        assign = assigns[name]
        corr_num = .0
        total_num = .0
        ids, labels = assign2label(assign)
        pid2label = {i:j for i, j in zip(ids, labels)}
        for pida, pidb in pairs:
            total_num += 1
            if (pid2label[pida] == pid2label[pidb]):
                corr_num += 1

        prec[name] = corr_num / total_num
    return prec


def cal_f1(prec, rec):
    return 2 * prec * rec / (prec + rec)


def pairwise_precision_recall_f1(preds, truths):
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j:
                if truths[i] == truths[j]:
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn

    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

# def offline_score():
#     assignments_train = json.load(open(assignments_train_path, 'r'))
#     Z = pkl.load(open(local_output_path, 'rb'))
#     metric = []
#     for k, v in assignments_train.items():
#         ids = [p['id'] for p in v]
#         X = [Z.get(x) for x in ids]
#         ids, labs = assign2label(v)

        # labels = clustering(X)
        # f1 = pairwise_precision_recall_f1(labels, labs)
        # print(f1)
        # metric.append(f1)
    # print('offline f1:', np.mean(metric))


# if __name__ == "__main__":
#     # 线下验证：
#     assignments_train = json.load(open(assignments_train_path, 'r'))

#     metric = []
#     for k, v in assignments_train.items():
#         ids, labs = assign2label(v)
#         model = AgglomerativeClustering(n_clusters=get_clusters(k))

#         X = make_input(ids)

#         model.fit(X)
#         f1 = pairwise_precision_recall_f1(model.label_, labs)
#         print(f1)
#         metric.append(f1)
#     print('offline f1:', np.mean(metric))
