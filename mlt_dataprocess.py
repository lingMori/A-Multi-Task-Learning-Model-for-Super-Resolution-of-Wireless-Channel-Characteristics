from __future__ import print_function
import os
import random
import sys
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from scipy import io
import errno
import copy


def seed_everything(args):
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def mat2npy():
    mat = io.loadmat('pro_data_tri.mat')
    print(mat['pro_data'].shape)
    print(mat['predot'].shape)
    np.save('TRI_data.npy', mat['pro_data'])
    np.save('TRI_predot.npy', mat['predot'])


# 针对联邦学习特化的数据集处理方式
def mat2npy_train():
    mat1 = io.loadmat('data/train_name.mat')  # 读取.mat的信息//'pro_data_tri.mat' 'pro_data' 'predot'
    np.save('train_name.npy', mat1['Data_final'])  # 二进制保存


def mat2npy_test():
    mat1 = io.loadmat('data/test_name.mat')  # 读取.mat的信息//'pro_data_tri.mat' 'pro_data' 'predot'
    np.save('test_name.npy', mat1['Data_final'])  # 二进制保存


def Load_ChannelCharacteristicsData(args):
    datapath = args.datapath
    test_ratio = args.test_ratio
    seed = args.seed
    if (args.datasplit_or_not == 1):
        channel_data = np.load(datapath)
        index_data = np.arange(channel_data.shape[0])

        _, _, channel_data_train, channel_data_test = train_test_split(index_data, channel_data, test_size=test_ratio,
                                                                       random_state=seed)
    else:
        channel_data_train = np.load(args.traindata)
        channel_data_test = np.load(args.testdata)

    channel_data_train = torch.from_numpy(channel_data_train)
    channel_data_test = torch.from_numpy(channel_data_test)
    channel_data_train = channel_data_train.type(torch.float32)
    channel_data_test = channel_data_test.type(torch.float32)

    # generate mask by LNLOS
    mask_value = args.maskvalue
    # mask_value = 1
    mask_train = torch.zeros(channel_data_train.shape[0], 1, 200, 200)  # 给空缺的区域给补全了
    mask_train = mask_train + channel_data_train[:, [6], :, :]
    mask_train[mask_train < 1] = -1;
    mask_train[mask_train > 0] = -mask_value;
    mask_train = -1 * mask_train

    mask_test = torch.zeros(channel_data_test.shape[0], 1, 200, 200)
    mask_test = mask_test + channel_data_test[:, [6], :, :]
    mask_test[mask_test < 1] = -1;
    mask_test[mask_test > 0] = -mask_value;
    mask_test = -1 * mask_test

    for i in np.arange(0, 199, args.scale):  # 跨幅度为2来进行赋值
        for j in np.arange(0, 199, args.scale):
            mask_train[:, :, i, j] = mask_value
            mask_test[:, :, i, j] = mask_value

    return channel_data_train, channel_data_test, mask_train, mask_test


def Dataugmentation(input_data):
    Rotate = transforms.functional.rotate
    Filp_H = transforms.RandomHorizontalFlip(p=1)
    Filp_V = transforms.RandomVerticalFlip(p=1)

    input_data_rotate90 = Rotate(input_data, 90)
    input_data_rotate180 = Rotate(input_data, 180)
    input_data_rotate270 = Rotate(input_data, 270)

    input_data_v = Filp_V(input_data)
    input_data_h = Filp_H(input_data)

    augmented_data = torch.cat(
        (input_data, input_data_rotate90, input_data_rotate180, input_data_rotate270, input_data_v, input_data_h),
        dim=0)

    return augmented_data


def INterplate(channel_data, scale, MODE):
    Output_data = F.interpolate(channel_data, scale_factor=1 / scale, mode=MODE, align_corners=True)  # size先缩小两倍
    Output_data = F.interpolate(Output_data, scale_factor=scale, mode=MODE, align_corners=True)  # size再扩大两倍

    for i in np.arange(0, 199, scale):
        for j in np.arange(0, 199, scale):
            Output_data[:, :, i, j] = channel_data[:, :, i, j]  # 保证0的数值不会被线性插值影响,进行一个覆盖

    return Output_data


def Dataset_Generator(Ground_truth, Input_data, mask, characteristic_index, target_index, batch_size, flag):
    if len([i for i in range(len(target_index)) if target_index[i] == 6]) > 0:
        ground_truth = Ground_truth[:, target_index, :, :]
        ground_truth[:, -1, :, :] = ground_truth[:, -1, :, :] + 1

    else:
        ground_truth = Ground_truth[:, target_index, :, :]

    input_data = Input_data[:, characteristic_index, :, :]

    # input_data = Input_data[:,target_index,:,:]
    # baselineloss = mlt_loss.L1loss_baseline()
    # std = mlt_loss.Std()
    # print('Baseline : ',baselineloss(input_data, ground_truth, mask)*180)
    # print('STD:',std(input_data,ground_truth,mask)*180)
    train_dataset = TensorDataset(input_data, ground_truth, mask)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=flag)

    return train_loader


fig = plt.figure()
ax0 = fig.add_subplot(121, title="train-loss")
ax1 = fig.add_subplot(122, title="test-loss")
fig_LOS = plt.figure()
ax00 = fig_LOS.add_subplot(121, title="train-loss")
ax11 = fig_LOS.add_subplot(122, title="ROC")


def draw_curve(epoch, me, rmse, name):  # 绘制Loss曲线的函数

    ax0.plot(epoch, me, 'bo-', label='train_loss')
    ax1.plot(epoch, rmse, 'bo-', label='test_loss')
    # ax0.legend()
    # ax1.legend()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', name))


def draw_curve_LOS(epoch, Train_LOSS, TPR_para, FPR_para, name):
    ax00.plot(epoch, Train_LOSS, 'bo-', label='train_loss')
    ax11.plot(FPR_para, TPR_para, 'bo-', label='ROC')
    # ax00.legend()
    # ax11.legend()
    fig_LOS.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', name))


class Record(object):  # 记录输出结果到txt中的函数
    def __init__(self, path=None):
        self.console = sys.stdout
        self.file = None
        if path is not None:  # 如果路径存在，
            mkdir_if_missing(os.path.dirname(path))
            self.file = open(path, 'w')  # 打开路径
        else:
            print('The path provided is wrong!')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(dir_path):  # 不需care
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def resume(model, resume_path):
    if not os.path.isfile(resume_path):
        print('                           ')
        print('can not load the model_wts!')
    else:
        model_wts = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(model_wts, strict=True)
        print('                          ')
        print('load model_wts successful!')
    return model
