import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


class Meter(object):
    """Computes and stores the average and current value"""
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



class TimeMeter(Meter):
    def __init__(self):
        Meter.__init__(self)

    def start(self):
        self.start_time = time.time()

    def record(self, n = 1):
        spent_time = time.time() - self.start_time
        self.update(spent_time, n)

def accuracy(output, target, topk=(1,)):
    if output.shape[1] == 1:
        output = (output.view(-1) > 0.5).long()
        correct = output.eq(target.view(-1))
        return [torch.sum(correct).float()/correct.shape[0] * 100]

    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def precision(c_matrix,ti):
    pre = c_matrix[ti,ti] / np.sum(c_matrix[:,ti])
    return pre

def recall(c_matrix,ti):
    recall = c_matrix[ti,ti] / np.sum(c_matrix[ti])
    return recall

def f_score(c_matrix,ti):
    pre = c_matrix[ti, ti] / np.sum(c_matrix[:, ti])
    recall = c_matrix[ti, ti] / np.sum(c_matrix[ti])
    score = 2 * pre * recall / (pre + recall)
    return score

def comfusion_matrix(preds, labels, c_num):
    confuse_m = np.zeros((c_num, c_num))
    for i in range(len(labels)):
        label = int(labels[i])
        pred = int(preds[i]) 
        confuse_m[label,pred] += 1
    return confuse_m
    
def auc_score(y_true,y_scores):
    if isinstance(y_true,torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    if isinstance(y_true, type([])):
        y_true = np.array(y_true).reshape((-1))

    if isinstance(y_scores, type([])):
        y_scores = np.array(y_scores).reshape((-1))
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = -1

    return auc