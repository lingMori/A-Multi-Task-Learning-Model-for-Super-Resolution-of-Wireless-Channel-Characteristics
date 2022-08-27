import re
import numpy as np
import torch
import torch.nn as nn
from model.attention.SEAttention import *
from model.attention.SKAttention import *

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
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.transConv_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.transConv_3 = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.Dropout = nn.Dropout(0.25)
        self.AveragePool = nn.AvgPool2d((2, 2))

    def forward(self, x):
        x1 = self.conv_1(x)
        print("x1 shape； ", x1.shape)

        x2 = self.conv_2(x1)
        print("x2 shape； ", x2.shape)
        x2 = self.Dropout(x2)

        x3 = self.conv_3(x2)
        print("x3 shape； ", x3.shape)
        x3 = self.Dropout(x3)

        y1 = self.transConv_1(x3 + x3)
        y1 = self.Dropout(y1)
        y2 = self.transConv_2(y1 + x2)
        y2 = self.Dropout(y2)
        y3 = self.transConv_3(y2 + x1)
        return y3


class SRCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SRCNN, self).__init__()

        self.convLayer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.convLayer(x)
        print(x.shape)
        return x


class SRCNN_EX(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SRCNN_EX, self).__init__()

        self.convLayer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True),
        )
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.convLayer(x)
        x1 = x + x1
        x2 = self.convLayer(x1)
        x2 = x2 + x1 + x
        x3 = self.convLayer2(x2)
        print(x3.shape)
        return x3


class FSRCNN(torch.nn.Module):
    def __init__(self, num_channels, upscale_factor, d=56, s=12):
        super(FSRCNN, self).__init__()
        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=5, stride=1, padding=0)
        self.prelu1 = nn.PReLU()

        # Shrinking（收缩）
        self.conv2 = nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0)
        self.prelu2 = nn.PReLU()

        # Non-linear Mapping
        self.conv3 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1)
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1)
        self.prelu6 = nn.PReLU()
        # Expanding
        self.conv7 = nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0)
        self.prelu7 = nn.PReLU()
        # Deconvolution（反卷积）
        self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=num_channels, kernel_size=9, stride=2,
                                            padding=4, output_padding=0)

    # 前向传播的过程
    def forward(self, x):  # x为输入数据
        out = self.prelu1(self.conv1(x))
        out = self.prelu2(self.conv2(out))
        out = self.prelu3(self.conv3(out))
        out = self.prelu4(self.conv4(out))
        out = self.prelu5(self.conv5(out))
        out = self.prelu6(self.conv6(out))
        out = self.prelu7(self.conv7(out))
        out = self.last_part(out)

        return out


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


features_in_hook = []
features_out_hook = []


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
        return x1

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
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=(3, 3), padding=1),

        )
        self.poweratio = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),

        )
        self.losn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
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


def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None


if __name__ == '__main__':
    # model = torchvision.models.resnet50()

    model = mltask(in_channels=7, out_channels=64)
    print(model)

    """
    layer_name = 'convLayer.2'
    for (name, module) in model.named_modules():
        print(name)
        if name == layer_name:
            module.register_forward_hook(hook=hook)
    """

    input = torch.randn(1, 7, 200, 200)
    thephi, poweratio, power, delay, los = model(input)

    
    # print(out.shape)
    """
    print(len(features_in_hook))  # 勾的是指定层的输入
    print(len(features_in_hook))  # 勾的是指定层的输出

    
    input = features_in_hook[0][0]
    output = features_out_hook[0]

    print(input.shape)
    print(output.shape)
    """
