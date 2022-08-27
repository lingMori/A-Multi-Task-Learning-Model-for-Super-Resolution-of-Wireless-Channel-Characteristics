import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from model.attention.SEAttention import *
from model.attention.SKAttention import *

sigmoid = nn.Sigmoid()


# model 3   简化版 模型
class Convd(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Convv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.finnal = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=(5, 5), padding=2)
        )

    def forward(self, x):
        x1 = self.Convv(x)
        x = x1 + x
        return self.finnal(x)


# model 4  骨干网络模型 分为 非los任务和 los任务
class Conv_Res_LOS(nn.Module):
    def __init__(self, in_channels):
        super(Conv_Res_LOS, self).__init__()
        self.Convv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.finalos = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=(5, 5), padding=2)
        )
        self.sfmx = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.Convv(x)
        x1 = x1 + x
        x1 = self.Convv(x1)
        x1 = x1 + x
        y = - torch.log(self.sfmx(self.finalos(x1)))
        return y  # b,3,200,200


class Conv_Res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Convv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.finnal = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), padding=2)
        )
        self.finalos = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=(5, 5), padding=2)
        )
        self.sfmx = nn.Softmax(dim=1)

    def forward(self, x, args):
        x1 = self.Convv(x)
        x1 = x1 + x
        x1 = self.Convv(x1)
        x1 = x1 + x

        if args.state == 0:
            y = self.finnal(x1)
        elif args.state == 1:
            y1 = self.finnal(x1)
            y2 = - torch.log(self.sfmx(self.finalos(x1)))
            y = (y1, y2)
        elif args.state == 2:
            y = - torch.log(self.sfmx(self.finalos(x1)))
        else:
            raise KeyError
        return y


# model 5 多任务学习模型
class mltask(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(mltask, self).__init__()
        self.net1 = Attention_2(in_channels, out_channels)
        self.net2 = net2_2(out_channels)
        self.weights = nn.Parameter(torch.ones(6).float())
        # self.net2 = Conv_Res_LOS(in_channels)

    def forward(self, x, args):
        y = self.net1(x)
        if args.period == 1:
            y = self.net2(y)
        else:
            y = self.net2(y.detach())
        return y

    def get_last_shared_layer(self):
        return self.net1


class net1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(net1, self).__init__()
        self.Convv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x1 = self.Convv(x)
        x1 = x1 + x
        x1 = self.Convv(x1)
        x1 = x1 + x
        # x1 = self.Convv(x1)
        # x1 = x1 + x
        return self.final(x1)


class net_1_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(net_1_2, self).__init__()
        '''
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        '''

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.AvgPool2d((2, 2)),
            nn.Dropout(0.25)
        )

        self.conv_2 = nn.Sequential(
            # nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.AvgPool2d((2, 2)),
            nn.Dropout(0.25)
        )

        self.conv_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.conv_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.conv_5 = nn.Sequential(
            nn.ConvTranspose2d(32, 7, kernel_size=(3, 3)),
            nn.BatchNorm2d(7),
            nn.ReLU()
        )


class net_1_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(net_1_3, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.transConv_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.transConv_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.transConv_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 7, kernel_size=(3, 3)),
            nn.BatchNorm2d(7),
            nn.ReLU()
        )

        self.Dropout = nn.Dropout(0.25)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x2 = self.Dropout(x2)
        x3 = self.conv_3(x2)
        x3 = self.Dropout(x3)
        y1 = self.transConv_1(x3 + x3)
        y1 = self.Dropout(y1)
        y2 = self.transConv_2(y1 + x2)
        y2 = self.Dropout(y2)
        y3 = self.transConv_3(y2 + x1)
        return y3


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()

        self.Convv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.attention = SEAttention(channel=in_channels, reduction=1)

    def forward(self, x):
        x1 = self.Convv(x)
        x1 = self.attention(x1)
        x1 = x1 + x
        x1 = self.Convv(x1)
        x1 = x1 + x
        # x1 = self.Convv(x1)
        # x1 = x1 + x
        return self.final(x1)


class Attention_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention_2, self).__init__()

        self.Convv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.Convv2 = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.attention = SEAttention(channel=64, reduction=7)

    def forward(self, x):
        x1 = self.Convv1(x)
        x1 = self.attention(x1)
        x1 = self.Convv2(x1)
        # x1 = self.attention(x1)
        x1 = x1 + x
        x1 = self.Convv1(x1)
        # x1 = self.attention(x1)
        x1 = self.Convv2(x1)
        # x1 = self.Convv(x1)
        x1 = x1 + x
        # x1 = self.Convv(x1)

        # x1 = x1 + x
        return self.final(x1)


class Attention_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention_3, self).__init__()

        self.Convv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.Convv2 = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.attention = SEAttention(channel=64, reduction=7)

    def forward(self, x):
        x1 = self.Convv1(x)
        # x1 = self.attention(x1)
        x1 = self.Convv2(x1)
        # x1 = self.attention(x1)
        x1 = x1 + x
        x1 = self.Convv1(x1)
        # x1 = self.attention(x1)
        x1 = self.Convv2(x1)
        # x1 = self.Convv(x1)
        x1 = x1 + x
        # x1 = self.Convv(x1)
        # x1 = x1 + x
        x1 = self.final(x1)
        return self.attention(x1)


class Attention_4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention_4, self).__init__()

        self.Convv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.Convv2 = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.attention = SEAttention(channel=64, reduction=7)

    def forward(self, x):
        x1 = self.Convv1(x)
        x1 = self.attention(x1)
        x1 = self.Convv2(x1)
        # x1 = self.attention(x1)
        x2 = x1 + x
        x1 = self.Convv1(x2)
        # x1 = self.attention(x1)
        x1 = self.Convv2(x1)
        # x1 = self.Convv(x1)
        x1 = x1 + x2
        # x1 = self.Convv(x1)

        # x1 = x1 + x
        return self.final(x1)


class Instead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Instead, self).__init__()

        self.Convv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.Convv2 = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.attention = SEAttention(channel=64, reduction=7)
        self.instead = nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x1 = self.Convv1(x)
        # x1 = self.attention(x1)
        x1 = self.instead(x1)
        x1 = self.Convv2(x1)
        # x1 = self.attention(x1)
        x1 = x1 + x
        x1 = self.Convv1(x1)
        # x1 = self.attention(x1)
        x1 = self.Convv2(x1)
        # x1 = self.Convv(x1)
        x1 = x1 + x
        # x1 = self.Convv(x1)
        return self.final(x1)


class net2(nn.Module):
    def __init__(self, in_channels):
        super(net2, self).__init__()
        self.power = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),

        )
        self.thetaphi = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=(3, 3), padding=1),

        )
        self.poweratio = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),

        )
        self.losn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=(3, 3), padding=1),

        )
        self.delay = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),

        )
        self.sfmx = nn.Softmax(dim=1)

    def forward(self, x):
        thephi = self.thetaphi(x)  # out_channel = 2
        poweratio = self.poweratio(x)  # out_channel = 1
        power = self.power(x)  # out_channel = 1
        delay = self.delay(x)  # out_channel = 1
        los = - torch.log(self.sfmx(self.losn(x)))  # out_channel = 1
        return thephi, poweratio, power, delay, los


class net2_2(nn.Module):
    def __init__(self, in_channels):
        super(net2_2, self).__init__()
        self.power = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),

        )
        self.thetaphi = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=(3, 3), padding=1)
        )
        self.poweratio = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),

        )
        self.losn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv2d(16, 3, kernel_size=(3, 3), padding=1),

        )
        self.delay = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),

        )
        self.sfmx = nn.Softmax(dim=1)

    def forward(self, x):
        thephi = self.thetaphi(x)  # out_channel = 2
        poweratio = self.poweratio(x)  # out_channel = 1
        power = self.power(x)  # out_channel = 1
        delay = self.delay(x)  # out_channel = 1
        los = - torch.log(self.sfmx(self.losn(x)))  # out_channel = 1
        return thephi, poweratio, power, delay, los
