import torch
import torch.nn as nn
from . import functional as fn


class fc_dnet(nn.Module):
    def __init__(self, in_planes, cluster):
        super(fc_dnet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_planes, cluster),
            nn.Linear(cluster, in_planes),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class VDN(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(VDN, self).__init__()
        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(1.0))
        self.a3 = nn.Parameter(torch.tensor(1.0))
        self.conv_w = nn.Conv2d(in_channels=num_classes, out_channels=1, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels=num_classes, out_channels=1, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_planes)
        self.fc = fc_dnet(in_planes, in_planes//16)
        self.f = 0
        self.w = 0
        self.w_grad = 0
        self.s = 0
        self.pd = None

    def forward(self, x, w, w_grad=None):
        b, n = x.shape
        weight = w.detach().t().view(n, -1, 1, 1) # NxCx1x1
        weight = self.conv_w(weight).view(-1)  # 512
        self.w = weight
        if w_grad is None:
            f_d = self.a1 * x + self.a2 * weight
        else:
            weightderi = w_grad.detach().t().view(n, -1, 1, 1)
            weightderi = self.conv_g(weightderi).view(-1)
            self.w_grad = weightderi
            f_d = self.a1 * x + self.a2 * weight + self.a3 * weightderi

        p_d = self.fc(f_d)
        x = p_d * x
        self.s = p_d
        p_d = 0.6*p_d+0.2
        self.pd = p_d
        x = fn.dropout(x, p_d, self.training)
        return x