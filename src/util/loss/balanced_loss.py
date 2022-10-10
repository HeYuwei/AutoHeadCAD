"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples"
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

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


class BalancedLoss(nn.Module):
    def __init__(self, no_of_classes, samples_per_cls, loss_type = 'focal', beta = 0.9999, gamma = 2.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.no_of_classes = no_of_classes
        # self.samples_per_cls = samples_per_cls
        # samples_per_cls = self.samples_per_cls
        # effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        # weights = (1.0 - self.beta) / np.array(effective_num)
        # weights = weights / np.sum(weights) * self.no_of_classes
        # self.weights = torch.tensor(weights).float()
        weights = BalancedLoss.cal_sample_weights(samples_per_cls, no_of_classes, beta)
        self.weights = torch.tensor(weights).float()
        self.weights = nn.Parameter(self.weights, requires_grad=False)

    @staticmethod
    def cal_sample_weights(samples_per_cls, no_of_classes, beta = 0.9999):
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        return weights.tolist()

    def forward(self, logits, labels):

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()
        weights = self.weights
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)
        loss_type = self.loss_type
        if loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
        elif loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss


if __name__ == '__main__':
    pass
    # no_of_classes = 5
    # logits = torch.rand(10,no_of_classes).float()
    # labels = torch.randint(0,no_of_classes, size = (10,))
    # beta = 0.9999
    # gamma = 2.0
    # samples_per_cls = [2,3,1,2,2]
    # loss_type = "focal"
    # cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
    # print(cb_loss)