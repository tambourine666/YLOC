import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nudging import *


class PCLoss(nn.Module):
    def __init__(self, num_classes, scale):
        super(PCLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.LongTensor([i for i in range(num_classes)]).cuda()
        self.scale = scale

    def forward(self, feature, target, proxy):
        '''
        feature: (N, dim)
        proxy: (C, dim)
        '''

        feature = F.normalize(feature, p=2, dim=1)
        pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))

        label = (self.label.unsqueeze(1) == target.unsqueeze(0))

        # print(target)

        pred_p = torch.masked_select(pred, label.transpose(1, 0))  # (N)   positive pair
        pred_p = pred_p.unsqueeze(1)

        pred_n = torch.masked_select(pred, ~label.transpose(1, 0)).view(feature.size(0),-1)  # (N, C-1) negative pair of anchor and proxy

        feature = torch.matmul(feature, feature.transpose(1, 0))
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)

        feature = feature * ~label_matrix
        feature = feature.masked_fill(feature < 1e-6, -np.inf)

        logits = torch.cat([pred_p, pred_n, feature], dim=1)
        label = torch.zeros(logits.size(0), dtype=torch.long).cuda()
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)
        return loss











