import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.resnet import resnet18
backbone_dict = {'resnet': resnet18}

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class MultiMILNet(nn.Module):
    def __init__(self, class_num=2, net_name='resnet', pool_mode='gated_attention', top_k = -1, **kwargs):
        super(MultiMILNet,self).__init__()  # call the initialization method of BaseModel

        self.backbone = backbone_dict[net_name](pretrained = True, **kwargs)
        self.pool_mode = pool_mode
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pool_mode = pool_mode
        self.top_k = top_k
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features_dict = {50: 2048, 34: 512, 18:512}
        bottleneck_dim_dict = {50: 256, 18: 256, 34:256}

        in_features = in_features_dict[kwargs['depth']]
        self.bottleneck_dim = bottleneck_dim_dict[kwargs['depth']]

        if self.pool_mode in ['attention', 'gated_attention']:
            self.attention_1 = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_gate = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_2 = nn.Linear(self.bottleneck_dim, 1)
            
            self.attention_1.apply(init_weights)
            self.attention_gate.apply(init_weights)
            self.attention_2.apply(init_weights)

        self.bottleneck = nn.Linear(in_features, self.bottleneck_dim)
        self.fc = nn.Linear(self.bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)
            
    def attention(self, x):
        x1 = self.attention_1(x)
        x_out = torch.tanh(x1)
        x2 = self.attention_2(x_out)
        return x2

    def gated_attention(self, x):
        x1 = self.attention_1(x)
        x_gate = torch.sigmoid(self.attention_gate(x))
        x_out = torch.tanh(x1) * x_gate
        x2 = self.attention_2(x_out)
        return x2

    def forward(self, input):
        tmp_shape = input.shape
        if len(tmp_shape) == 5:
            n_shape = [tmp_shape[0] * tmp_shape[1], tmp_shape[2], tmp_shape[3], tmp_shape[4]]
            input = input.view(n_shape)

        x = self.backbone(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(self.bottleneck(x))

        if len(tmp_shape) != 5:
            return y

        x = x.view(tmp_shape[0], tmp_shape[1], -1)
        y = y.view(tmp_shape[0], tmp_shape[1], -1)
        slice_y = y

        if self.top_k > 0:
            weights = F.softmax(y, dim=-1)[..., 1]
            max_inds = torch.argsort(weights, descending=True)[:, :self.top_k]
            x = torch.gather(x, dim = 1, index=max_inds)
            y = torch.gather(y, dim = 1, index=max_inds)

        elif self.pool_mode == 'gated_attention':
            weights = self.gated_attention(x)
            normalized_weights = F.softmax(weights, dim=1)
            scan_y = torch.sum(normalized_weights * x, dim=1)
            scan_y = self.bottleneck(scan_y)
            scan_y = self.fc(scan_y)

        return slice_y, scan_y, weights