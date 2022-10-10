import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000, size_average = True, smooth_ratio = 1, alpha = None, device = 0):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.size_average = size_average
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
        self.smooth_ratio = smooth_ratio

        self.alpha = alpha
        if alpha is not None:
            if isinstance(alpha,(float,int)):
                self.alpha = torch.Tensor([alpha, 1])
            if isinstance(alpha,list):
                self.alpha = torch.Tensor(alpha)
            self.size_average = size_average
            self.alpha = self.alpha.cuda(device)

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((self.smooth_ratio-(Yg**self.q))/self.q)*self.weight[indexes] - ((self.smooth_ratio-(self.k**self.q))/self.q)*self.weight[indexes]

        if self.alpha is not None:
            if self.alpha.type()!= logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0,targets.data.view(-1))
            loss = loss * Variable(at)

        if self.size_average:
            loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)
