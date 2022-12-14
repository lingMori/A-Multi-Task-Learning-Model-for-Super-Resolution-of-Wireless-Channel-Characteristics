from re import L
import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random


class traloss0(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none')
        # self.loss_v2 = nn.MSELoss(reduction='none')

    def forward(self, yhat, gtruth, mask):
        """
        函数功能：计算训练过程，真实值和预测值之间的l1loss，详情计算公式见论文
        :param yhat: 预测值
        :param gtruth: 真实值
        :param mask: 蒙版
        :return: l1loss损失值
        """
        yhat = torch.mul(yhat, mask)
        gtruth = torch.mul(gtruth, mask)
        predot = torch.sum(mask, dim=[2, 3])   # 把batch * channel * 200 * 200的矩阵压缩成(batch, channel, 1)的矩阵
        ans = torch.sum(self.loss(yhat, gtruth), dim=[2, 3]) / predot
        ans = torch.mean(ans, dim=0)
        return ans  # c


class tesloss0(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, yhat, gtruth, mask):
        """
        函数功能：计算测试过程，真实值和预测值之间的l1loss，详情计算公式见论文
        :param yhat: 预测值
        :param gtruth: 真实值
        :param mask: 蒙版
        :return: l1loss损失值
        """
        yhat = torch.mul(yhat, mask)
        gtruth = torch.mul(gtruth, mask)
        predot = torch.sum(mask, dim=[2, 3])
        ans = torch.sum(self.loss(yhat, gtruth), dim=[2, 3]) / predot
        ans = torch.mean(ans, dim=0)
        return ans  # c


class traloss2(nn.Module):
    def __init__(self):
        super(traloss2, self).__init__()

    def forward(self, y_hat, label, mask):
        """
        函数功能：计算训练过程，针对los/Nlos的损失函数
        :param y_hat: 预测值
        :param label: 真实值
        :param mask: 蒙版
        :return: 损失值
        """
        label = label.long()  # b 1 200 200
        lossmatrix = torch.gather(y_hat, dim=1, index=label)  # b 1 200 200
        lossmatrix = torch.mul(lossmatrix, mask)
        predot = torch.sum(mask, dim=[1, 2, 3])
        ans = torch.sum(lossmatrix, dim=[1, 2, 3]) / predot
        ans = torch.mean(ans, dim=0)
        return ans


class tesloss2(nn.Module):
    def __init__(self):
        super(tesloss2, self).__init__()

    def forward(self, y_hat, label, mask):  # yhat n,3,200,200     mask n,1,200 200
        """
        函数功能：计算测试过程，针对los/Nlos计算TPR和FPR
        :param y_hat: 预测值
        :param label: 真实值
        :param mask: 蒙版
        :return: 损失值
        """
        premat = torch.argmin(y_hat, dim=1)  # b 1 200 200
        P_pre = torch.mul(torch.where(premat == 0, 1, 0), mask)
        N_pre = torch.mul(torch.where(premat == 1, 1, 0), mask)
        P_label = torch.mul(torch.where(label == 0, 1, 0), mask)
        N_label = torch.mul(torch.where(label == 1, 1, 0), mask)
        TPR = torch.sum(torch.mul(P_pre, P_label), dim=[1, 2, 3]) / torch.sum(P_label, dim=[1, 2, 3])
        FPR = torch.sum(torch.mul(P_pre, N_label), dim=[1, 2, 3]) / torch.sum(N_label, dim=[1, 2, 3])
        # para1 = torch.sum(torch.mul(N_pre,N_label))/torch.sum(N_label)   # 1-FPR
        # para2 = torch.sum(torch.mul(N_pre,P_label))/torch.sum(P_label)   # 1-TPR
        TPR = torch.mean(TPR, dim=0);
        FPR = torch.mean(FPR, dim=0)
        return TPR, FPR


# 用于各个特征的（单个输入，不可同时使用） 插值方法的loss结果 统计
class L1loss_baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, y_hat, g_truth, mask):
        y_hat = torch.mul(y_hat, mask)
        g_truth = torch.mul(g_truth, mask)
        predot = torch.sum(mask, dim=[1, 2, 3])
        ans = torch.sum(self.loss(y_hat, g_truth), dim=[1, 2, 3]) / predot
        return torch.mean(ans, dim=0)


class Std(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, y_hat, g_truth, mask):
        """
        函数功能：计算真实值和预测值之间的标准差（STDE）
        :param y_hat: 预测值
        :param g_truth: 真实值
        :param mask: 蒙板
        :return: std标准差
        """
        y_hat = torch.mul(y_hat, mask)  # b c
        g_truth = torch.mul(g_truth, mask)
        predot = torch.sum(mask, dim=[1, 2, 3]).view(mask.shape[0], 1, 1, 1)  # b 1 1 1
        loss = y_hat - g_truth  # b c 200 200
        mean = torch.div(torch.sum(loss, dim=[2, 3], keepdim=True), predot)  # b c 1 1  b 1 1 1
        para1 = torch.mul((loss - mean) ** 2, mask)  # b c 200 200
        para2 = torch.sqrt(torch.sum(para1, dim=[2, 3], keepdim=True) / predot)  # b c 1 1
        para2 = torch.mean(para2, dim=0).view(y_hat.shape[1])
        return para2


class bploss(nn.Module):
    def __init__(self, length, args):
        super().__init__()
        self.length = length
        self.coefficient1 = nn.Parameter(torch.zeros(1))
        self.coefficient2 = nn.Parameter(torch.zeros(1))
        self.coefficient3 = nn.Parameter(torch.zeros(1))
        self.coefficient4 = nn.Parameter(torch.zeros(1))
        self.coefficient5 = nn.Parameter(torch.zeros(1))
        self.coefficient6 = nn.Parameter(torch.zeros(1))

        self.coefficient11 = nn.Parameter(torch.zeros(1))
        self.coefficient22 = nn.Parameter(torch.zeros(1))
        self.coefficient33 = nn.Parameter(torch.zeros(1))
        self.coefficient44 = nn.Parameter(torch.zeros(1))
        self.coefficient55 = nn.Parameter(torch.zeros(1))

    def forward(self, loss, std):
        """
        函数功能：Uncertainty weight，将输入的多任务的loss和std之间按照不同的权重进行再合并
        :param loss: 输入合并的多任务loss
        :param std: 输入合并的多任务std
        :return: 调整后的loss
        """
        self.coefficient = torch.cat(
            (self.coefficient1, self.coefficient2, self.coefficient3, self.coefficient4, self.coefficient5,
             self.coefficient6), dim=0)
        bploss = torch.sum(torch.mul(loss, 1 / (2 * torch.exp(self.coefficient)))) + torch.sum(self.coefficient)

        self.coefficient_part2 = torch.cat(
            (self.coefficient11, self.coefficient22, self.coefficient33, self.coefficient44, self.coefficient55), dim=0
        )
        stdloss = torch.sum(torch.mul(std, 1 / (2 * torch.exp(self.coefficient_part2)))) + torch.sum(self.coefficient_part2)
        # bploss = torch.sum(loss)
        Loss = stdloss + bploss
        return Loss