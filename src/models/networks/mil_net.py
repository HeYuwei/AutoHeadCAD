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

class MILNet(nn.Module):

    def __init__(self, net_name='resnet', bottleneck_dim=256, class_num=2, pool_mode='gated_attention', topk=-1, **kwargs):
        super(MILNet,self).__init__()  # call the initialization method of BaseModel

        self.backbone = backbone_dict[net_name](**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.in_features = 512
        self.pool_mode = pool_mode
        self.topk = topk
        self.bottleneck_dim = bottleneck_dim

        if self.pool_mode == 'gated_attention':
            self.attention_1 = nn.Linear(self.in_features, self.bottleneck_dim)
            self.attention_gate = nn.Linear(self.in_features, self.bottleneck_dim)
            self.attention_2 = nn.Linear(self.bottleneck_dim, 1)
            
            self.attention_1.apply(init_weights)
            self.attention_gate.apply(init_weights)
            self.attention_2.apply(init_weights)
        
        self.bottleneck = nn.Linear(self.in_features, self.bottleneck_dim)
        self.fc = nn.Linear(self.bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)

    def gated_attention(self, x):
        x1 = self.attention_1(x)
        x_gate = torch.sigmoid(self.attention_gate(x))
        x_out = torch.tanh(x1) * x_gate
        x2 = self.attention_2(x_out)
        return x2

    def forward(self, input):
        x = self.backbone(input) 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(self.bottleneck(x))
        orig_y = y

        if self.topk > 0:
            weights = F.softmax(y, dim=-1)[:, 1]
            _, max_inds = weights.topk(self.topk)
            x = x[max_inds]
            y = y[max_inds]

        if self.pool_mode == 'max':
            scores = F.softmax(y, dim=1)[:, 1]
            max_ind = torch.argmax(scores).item()
            weights = torch.zeros(len(scores), 1)
            weights[max_ind][0] = 1.0
            total_y = self.fc(self.bottleneck(x[max_ind:max_ind+1]))
        elif self.pool_mode == 'average':
            weights = torch.zeros(y.shape[0], 1)
            weights[:] = 1.0 / y.shape[0]
            total_y = torch.sum(x, dim=0).unsqueeze(0) / y.shape[0]
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'exp_attn':
            weights = F.softmax(y, dim=-1)[:, 1]
            _, max_inds = weights.topk(3)
            x = x[max_inds]
            y = y[max_inds]
            normalized_weights = F.softmax(weights, dim=0).detach().view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'gated_attention':
            weights = self.gated_attention(x)
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))

        return orig_y, total_y, weights