# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn


def conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels)
    )


def conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1, 1),
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_channels)
    )


class BaseModel(torch.nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()

        self.feat_dim = opt["base_feat_dim"]
        self.output_dim = opt["base_out_dim"]

        self.conv1 = nn.Conv1d(in_channels=self.feat_dim, out_channels=self.output_dim * 2, kernel_size=3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.output_dim * 2, out_channels=self.output_dim, kernel_size=3,
                               stride=1, padding=1)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class TEM(torch.nn.Module):
    def __init__(self, opt):
        super(TEM, self).__init__()

        self.feat_dim = opt["tem_feat_dim"]
        self.c_hidden = opt["tem_hidden_dim"]
        self.output_dim = 2

        self.conv1 = nn.Conv1d(in_channels=self.feat_dim, out_channels=self.c_hidden, kernel_size=3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.c_hidden, out_channels=self.c_hidden, kernel_size=3, stride=1,
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.c_hidden, out_channels=self.output_dim, kernel_size=1, stride=1,
                               padding=0)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(0.1 * self.conv3(x))
        return x


class BM_layer(torch.nn.Module):
    def __init__(self, bm_mask):
        super(BM_layer, self).__init__()
        self.bm_mask = bm_mask  # T,n,D,T
        self.temporal_dim, self.num_sample_point, self.duration, _ = bm_mask.shape
        self.bm_mask = bm_mask.view(self.temporal_dim, -1).cuda()

    def forward(self, data):  # N, C, T
        input_size = data.size()
        x_view = data.view(-1, input_size[-1])
        # feature [bs*C, T]
        # bm_mask [T, N*D*T]
        # out     [bs, C, N, D, T]
        result = torch.matmul(x_view, self.bm_mask)
        return result.view(input_size[0], input_size[1], self.num_sample_point, self.duration, self.temporal_dim)


class PEM(torch.nn.Module):
    def __init__(self, opt):
        super(PEM, self).__init__()
        # self.temporal_scale = opt["temporal_scale"]
        self.feat_dim = opt["pem_feat_dim"]
        self.hidden_dim = opt["pem_hidden_dim"]
        self.u_ratio_m = opt["pem_u_ratio_m"]
        self.u_ratio_l = opt["pem_u_ratio_l"]
        self.sample_num = opt["pem_sample_num"]
        self.bm_mask = torch.Tensor(opt["bm_mask"])
        self.output_dim = 2
        self.BMlayer = BM_layer(self.bm_mask)
        self.conv1 = nn.Conv3d(in_channels=self.feat_dim, out_channels=self.feat_dim * 4,
                               kernel_size=(self.sample_num, 1, 1),
                               stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.feat_dim * 4, out_channels=self.feat_dim, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=self.feat_dim, out_channels=self.feat_dim, kernel_size=3, stride=1,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.feat_dim, out_channels=self.output_dim, kernel_size=1, stride=1,
                               padding=0)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = self.BMlayer(x)
        x = F.relu(self.conv1(x)).squeeze(2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(0.1 * self.conv4(x))
        return x


class BMN(nn.Module):
    def __init__(self, opt):
        super(BMN, self).__init__()
        self.best_loss = 100000
        self.base_model = BaseModel(opt)
        self.tem_model = TEM(opt)
        self.pem_model = PEM(opt)

    def forward(self, x):
        base_feature = self.base_model(x)
        start_end = self.tem_model(base_feature)
        confidence_map = self.pem_model(base_feature)
        return start_end, confidence_map
