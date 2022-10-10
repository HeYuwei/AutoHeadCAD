from util.basic import *
import collections
import torch.nn as nn
import torch.nn.functional as F
from util.loss import FocalLoss, TruncatedLoss, BalancedLoss
from .ctbase_model import CTBaseModel


class TrainScanModel(CTBaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser = CTBaseModel.modify_commandline_options(parser)
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        '''The weights used to balance positive and negative samples'''
        pos_samples_num = len(self.opt.dataset.dataset.dataset_1_json)
        neg_samples_num = len(self.opt.dataset.dataset.dataset_0_json)
        sample_weights = BalancedLoss.cal_sample_weights(samples_per_cls=[pos_samples_num, neg_samples_num],
                                                         no_of_classes=opt.num_classes)

        '''Truncated Loss: Used for positive samples'''
        if 'truncated' in self.opt.loss_type:
            self.criterion_truncated = TruncatedLoss(trainset_size=len(self.opt.dataset.dataset.dataset_0_json) + len(self.opt.dataset.dataset.dataset_1_json), size_average= opt.loss_average,
                                                     alpha=sample_weights)
            self.criterion_truncated.cuda(self.opt.gpu_ids[0])

        '''Focal Loss: Used for negative samples. It is equivalent to Cross-Entropy loss here'''
        if 'focal' in self.opt.loss_type:
            self.criterion_focal = FocalLoss(gamma=1, alpha=sample_weights,
                                             device=self.opt.gpu_ids[0])

    def set_input(self, data):
        super().set_input(data)
        self.input = torch.stack(self.input).cuda(self.opt.gpu_ids[0])
        self.index = torch.Tensor(self.input_id).cuda(self.opt.gpu_ids[0]).long()

    def setup(self, opt):
        super().setup(opt)

    def forward(self):
        self.slice_y, self.y, _ = self.net_res(self.input)

        '''During evaluation, the scan-level probability can also be obtained
         by selecting the the max slice-level probability'''
        if self.opt.l_state != 'train' and self.opt.valid_with_single_slice:
            max_inds = torch.argmax(self.slice_y[..., 1], dim=-1)
            self.y = torch.gather(self.slice_y, dim=1, index=max_inds.view(-1, 1))
            self.y = self.y.view(-1, self.opt.num_classes)

        self.series_y = self.y.detach()
        tmp_y = F.softmax(self.series_y, dim=-1)
        self.score = tmp_y[:, 1]
        self.series_score = self.score
        self.bag_ys = self.slice_y.detach()
        tmp_ys = F.softmax(self.bag_ys, dim=-1)
        self.bag_scores = tmp_ys[:, -1]

    def cal_loss(self):
        if self.opt.l_state != 'train':
            self.loss_c = nn.MSELoss()(self.score, self.label.float())
            return

        if 'truncated' in self.opt.loss_type and 'focal' in self.opt.loss_type:
            inds_0 = torch.where(self.label == 0)[0]
            inds_1 = torch.where(self.label == 1)[0]

            self.loss_c = 0
            if inds_0.shape[0]:
                self.loss_c += self.criterion_focal(self.y[inds_0], self.label[inds_0])
            if inds_1.shape[0]:
                self.loss_c = self.criterion_truncated(self.y[inds_1], self.label[inds_1], self.index[inds_1])
            if self.opt.epoch % 2 == 0:
                self.criterion_truncated.update_weight(self.y[inds_1], self.label[inds_1], self.index[inds_1])

        elif self.opt.loss_type == 'truncated':
            self.loss_c = self.criterion_truncated(self.y, self.label, self.index)
            if self.opt.epoch % 2 == 0:
                self.criterion_truncated.update_weight(self.y, self.label, self.index)

        elif self.opt.loss_type == 'focal':
            self.loss_c = self.criterion_focal(self.y, self.label)

    def get_parameters(self):
        parameter_list = [{"params": self.net_res.backbone.parameters(), "lr_mult": 1, 'decay_mult': 1},
                          {"params": self.net_res.fc.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult},
                          {"params": self.net_res.bottleneck.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult}]

        if self.opt.pool_mode in ['attention', 'gated_attention']:
            parameter_list.append({"params":self.net_res.attention_1.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})
            parameter_list.append({"params":self.net_res.attention_gate.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})
            parameter_list.append({"params":self.net_res.attention_2.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})

        return parameter_list
