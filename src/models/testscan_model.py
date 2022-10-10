from util import *
import torch.nn as nn
import torch.nn.functional as F
from util.loss import FocalLoss
from .ctbase_model import CTBaseModel


class TestScanModel(CTBaseModel):
    """
    Model class for testing.
    """
    
    def __init__(self, opt):
        assert opt.method_name in ['mil']
        super().__init__(opt)
        for l_state in ['train','valid','test']:
            setattr(self, l_state+'_loss_names', [])
        self.train_loss_names.append('series_c')
        self.buffer_g_weights = [] # for MIL model only

    def train(self):
        for name in self.net_names:
            net = getattr(self, 'net_' + name)
            net.train()

    def setup(self, opt):
        super().setup(opt)
        
    def _forward(self, detach):
        batch = len(self.input)
        self.y = []
        self.bag_ys = []
        self.series_y = []
        self.bag_scores = []
        self.score = []
        self.weights = []
        for i in range(batch):
            bag_ys, y, weights = self.net_res(self.input[i].cuda(self.gpu_ids[0]))
            # visualization: calculate grad by backward()
            bag_out = F.softmax(bag_ys, dim=1)
            y_out = F.softmax(y, dim=1)
            if self.opt.l_state != 'train' and self.opt.visualize and self.opt.vis_method in ['gradcam']:
                self.zero_grad()
                self.vis_method.cal_grad(bag_out, 1)
            series_y = torch.tensor(y)
            if detach:
                bag_ys = bag_ys.detach()
                y = y.detach()
            bag_scores = bag_out.detach().cpu()[:, 1].view(-1)
            score = y_out.detach().cpu()[:, 1].view(-1)
            self.y.append(y)
            self.bag_ys.append(bag_ys)
            self.series_y.append(series_y)
            self.bag_scores.append(bag_scores)
            self.score.append(score)
            self.weights.append(weights)
        
        self.y = torch.stack(self.y, dim=0).mean(axis=0, keepdim=False)
        self.bag_ys = torch.stack(self.bag_ys, dim=0).mean(axis=0, keepdim=False)
        self.series_y = torch.stack(self.series_y, dim=0).mean(axis=0, keepdim=False)
        self.bag_scores = torch.stack(self.bag_scores, dim=0).mean(axis=0, keepdim=False)
        self.score = torch.cat(self.score, dim=0).mean(axis=0, keepdim=True)
        self.series_score = self.score
        self.weights = torch.stack(self.weights, dim=0).mean(axis=0, keepdim=False)

    def forward(self):
        if self.opt.l_state == 'train':
            assert self.opt.dataset_mode in ['vis', 'scan']
            self._forward(False)
        else:
            assert self.opt.v_dataset_mode in ['vis', 'scan', 'cq500']
            self._forward(True)
    
    def backward(self):
        self.update_metrics('local')
        total_loss = 0
        for name in self.loss_names:
            loss = getattr(self,'loss_' + name) / self.opt.grad_iter_size
            total_loss += loss
        total_loss.backward()

    def cal_loss(self):
        if self.opt.l_state == 'train':
            if self.opt.loss_type == 'focal':
                self.loss_series_c = FocalLoss(gamma=self.opt.focal_gamma, alpha=self.opt.focal_alpha, device=self.opt.gpu_ids[0])(
                    self.y.cuda(self.gpu_ids[0]), self.label.cuda(self.gpu_ids[0]).long())
            else:
                self.loss_series_c = nn.CrossEntropyLoss()(self.y, self.label.long())

    def stat_info(self):
        super().stat_info()
        self.buffer_g_weights.extend(self.weights.cpu().view(-1).tolist())