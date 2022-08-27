from __future__ import print_function
import os
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, TensorDataset
import mlt_dataprocess
import mlt_model
import argparse
import mlt_train0
import mlt_train1
import mlt_train3
import mlt_train2
import sys

# sys.path.append("./")
work_dir = os.path.dirname(os.path.abspath(__file__))
Data = os.path.join(work_dir,'data')
N_LOS_switch = False  # 这里什么意思
state = 0
torch.cuda.set_device(0)
parser = argparse.ArgumentParser(description='Mltask prediction & classification')  # 解析器
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--datapath', type=str, default=os.path.join(Data, "all_all_train.npy"))
parser.add_argument('--datasplit_or_not', type=bool, default=1) 
parser.add_argument('--traindata', type=str, default=os.path.join(Data, "only_SH.npy"))
parser.add_argument('--testdata', type=str, default=os.path.join(Data, "without_SH.npy"))
parser.add_argument('--period', type=int, default=1)
parser.add_argument('--scale', type=int, default=8)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--step_size', type=int, default=30)
parser.add_argument('--aug_switch', type=bool, default=True)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--gamma', type=float, default=0.7)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--maskvalue', type=float, default=1e-5)
parser.add_argument('--epochs_P1', type=int, default=5)
parser.add_argument('--epochs_P2', type=int, default=5)

# **************************修改关键参数******************************
# keyinfo = 'Task10 MTL All-All-2-noite-nores-noaug'
keyinfo = 'scale=8'  # 保存的结果文件名称
characteristic_index = np.array([0, 1, 2, 3, 4, 5, 6])
target_index = np.array([1, 2, 3, 4, 5, 6])

# channel characteristics : height K phi theta p t los
if len([i for i in range(len(target_index)) if target_index[i] == 6]) > 0:  # 这一小快是判断target_index里面是否有N_LOS
    N_LOS_switch = True
    state = 1
    if len(target_index) == 1:  # 有N_los但是同时target中只有N_Los
        state = 2
parser.add_argument('--characteristic_index', type=np.ndarray, default=characteristic_index)
parser.add_argument('--target_index', type=np.ndarray, default=target_index)
parser.add_argument('--N_LOS_switch', type=bool, default=N_LOS_switch)
parser.add_argument('--state', type=int, default=state)
parser.add_argument('--weight_lastname', type=str, default=keyinfo + '.pth')
parser.add_argument('--logname', type=str, default=keyinfo + '.txt')
parser.add_argument('--jpgname', type=str, default=keyinfo + '.jpg')
parser.add_argument('--csvname', type=str, default=keyinfo + '.csv')
parser.add_argument('--argsname', type=str, default='args' + keyinfo + '.txt')
parser.add_argument('--wight_lastname', type=str, default=keyinfo + '.pth')
args = parser.parse_args()

# random seed set

mlt_dataprocess.seed_everything(args)

# 第一次数据转化时需要
# mlt_dataprocess.mat2npy_train()

# load & mask
# print(args.traindata)
channel_data_train, channel_data_test, mask_train, mask_test = mlt_dataprocess.Load_ChannelCharacteristicsData(args)
print('Training set shape:', channel_data_train.shape)
print('Test set shape:', channel_data_test.shape)

# interplate     线性插值
Input_data_train = mlt_dataprocess.INterplate(channel_data_train, args.scale, 'bilinear')
Input_data_test = mlt_dataprocess.INterplate(channel_data_test, args.scale, 'bilinear')

# augumentation    预置（数据增强？？？应该是的,里面都是旋转这种不会改变相对关系的操作）
if args.aug_switch == 1:
    channel_data_train = mlt_dataprocess.Dataugmentation(channel_data_train)
    Input_data_train = mlt_dataprocess.Dataugmentation(Input_data_train)
    mask_train = mlt_dataprocess.Dataugmentation(mask_train)

print(channel_data_train.shape, channel_data_test.shape, Input_data_train.shape, Input_data_test.shape,
      mask_train.shape, mask_test.shape)

# wrap

test_loader = mlt_dataprocess.Dataset_Generator(channel_data_test, Input_data_test, mask_test,
                                                characteristic_index, target_index, args.batch_size, flag='False')
train_loader = mlt_dataprocess.Dataset_Generator(channel_data_train, Input_data_train, mask_train,
                                                 characteristic_index, target_index, args.batch_size, flag='True')

print('Length of Training set:', len(train_loader), 'Length of Test set:', len(test_loader))

# model initialization
# model = modelxp.Discriminator().to(device)
# model = resnet50.ResNet50(len(characteristic_index)).to(device)

if args.state == 1:  # 话说这里有必要用两个吗？不都是一样的意思吗。。。
    model = mlt_model.mltask(in_channels=len(characteristic_index), out_channels=64).to(args.device)
else:
    model = mlt_model.mltask(in_channels=len(characteristic_index), out_channels=64).to(args.device)  # model的shape是一个1

# model parameters and computation statistics

# print()
# dummy_input = torch.rand(1,7,200,200).to(device)
# flops,params = get_model_complexity_info(model,(7,200,200),as_strings=True,print_per_layer_stat=True)
# print('FLOPs:',flops,'params',params)
# print(summary(model,(7,200,200)))
# print()

# resume weights trained before
# model = vit_dataprocess.resume(model,os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'temp.pth'))

# train
if state == 0:
    print('train0')
    mlt_train0.train(model, args, train_loader, test_loader)
elif state == 1:
    print('train1')
    # mlt_train1.train(model, args, train_loader, test_loader)
    mlt_train3.train(model, args, train_loader, test_loader)
elif state == 2:
    print('train2')
    mlt_train2.train(model, args, train_loader, test_loader)
else:
    raise KeyError

# save args_information
argsDict = args.__dict__
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', args.argsname), 'w') as f:
    f.writelines('--------------start---------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ': ' + str(value) + '\n')
    f.writelines('---------------end----------------')
