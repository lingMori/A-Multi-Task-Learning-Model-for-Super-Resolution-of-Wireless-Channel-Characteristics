from __future__ import print_function
import os
import numpy as np
import torch.utils.data
import mlt_dataprocess
import mlt_model
import argparse
import mlt_train0
import mlt_train3
import mlt_train2

work_dir = os.path.dirname(os.path.abspath(__file__))   # 获取当前文件目录
Data = os.path.join(work_dir, 'data')                   # 将数据路径添加入主路径当中（适配windows和Linux）
N_LOS_switch = False
state = 0
torch.cuda.set_device(0)
parser = argparse.ArgumentParser(description='Mltask prediction & classification')  # 解析器
# 检测是否有显卡，无限卡调用cpu
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--datapath', type=str, default=os.path.join(Data, "all_all_train.npy"))   # 数据集名称
parser.add_argument('--datasplit_or_not', type=bool, default=1)                                # 是否采用联邦训练
parser.add_argument('--traindata', type=str, default=os.path.join(Data, "only_SH.npy"))        # 训练集
parser.add_argument('--testdata', type=str, default=os.path.join(Data, "without_SH.npy"))      # 测试集
parser.add_argument('--period', type=int, default=1)                                           # 选择训练模式
parser.add_argument('--scale', type=int, default=8)                                            # 超分尺度
parser.add_argument('--test_ratio', type=float, default=0.2)                                   # 测试机比例（占整体数据集）
parser.add_argument('--seed', type=int, default=42)                                            # 随机种子
parser.add_argument('--batch_size', type=int, default=1)                                       # 训练数量/batch
parser.add_argument('--step_size', type=int, default=30)                                       # 步长
parser.add_argument('--aug_switch', type=bool, default=True)                                   # 是否采用数据增强的方式
parser.add_argument('--lr', type=float, default=1e-5)                                          # 学习率
parser.add_argument('--gamma', type=float, default=0.7)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--maskvalue', type=float, default=1e-5)                                   # 蒙版的数值
parser.add_argument('--epochs_P1', type=int, default=5)                                        # 阶段一训练回数
parser.add_argument('--epochs_P2', type=int, default=5)                                        # 阶段二训练回数

# **************************修改关键参数******************************

keyinfo = 'scale=8'  # 保存的结果文件名称
characteristic_index = np.array([0, 1, 2, 3, 4, 5, 6])   # 输入的特征选择
# channel characteristics : height K phi theta p t los
target_index = np.array([1, 2, 3, 4, 5, 6])              # 超分目标的特征选择
# target_index : K phi theta p t los

if len([i for i in range(len(target_index)) if target_index[i] == 6]) > 0:  # 这一小快是判断target_index里面是否有N_LOS
    N_LOS_switch = True
    state = 1
    if len(target_index) == 1:  # 有N_los但是同时target中只有N_Los
        state = 2
parser.add_argument('--characteristic_index', type=np.ndarray, default=characteristic_index)
parser.add_argument('--target_index', type=np.ndarray, default=target_index)
parser.add_argument('--N_LOS_switch', type=bool, default=N_LOS_switch)
parser.add_argument('--state', type=int, default=state)
parser.add_argument('--weight_lastname', type=str, default=keyinfo + '.pth')          # 保存权重文件
parser.add_argument('--logname', type=str, default=keyinfo + '.txt')                  # 保存训练记录
parser.add_argument('--jpgname', type=str, default=keyinfo + '.jpg')                  # 保存生成图像
parser.add_argument('--csvname', type=str, default=keyinfo + '.csv')                  # 保存训练结果
parser.add_argument('--argsname', type=str, default='args' + keyinfo + '.txt')        # 保存参数配置
parser.add_argument('--wight_lastname', type=str, default=keyinfo + '.pth')
args = parser.parse_args()

# random seed set

mlt_dataprocess.seed_everything(args)

# 第一次数据转化时需要
# mlt_dataprocess.mat2npy_train()
# mlt_dataprocess.mat2npy_test()

# load & mask
# 数据集的装载
channel_data_train, channel_data_test, mask_train, mask_test = mlt_dataprocess.Load_ChannelCharacteristicsData(args)
print('Training set shape:', channel_data_train.shape)
print('Test set shape:', channel_data_test.shape)

# interplate     对训练集和测试及线性插值
Input_data_train = mlt_dataprocess.INterplate(channel_data_train, args.scale, 'bilinear')
Input_data_test = mlt_dataprocess.INterplate(channel_data_test, args.scale, 'bilinear')

# augumentation    训练集的数据增强处理
if args.aug_switch == 1:
    channel_data_train = mlt_dataprocess.Dataugmentation(channel_data_train)
    Input_data_train = mlt_dataprocess.Dataugmentation(Input_data_train)
    mask_train = mlt_dataprocess.Dataugmentation(mask_train)

print(channel_data_train.shape, channel_data_test.shape, Input_data_train.shape, Input_data_test.shape,
      mask_train.shape, mask_test.shape)

# wrap 将处理好的数据进行打包，包装成Dataset格式的数据

test_loader = mlt_dataprocess.Dataset_Generator(channel_data_test, Input_data_test, mask_test,
                                                characteristic_index, target_index, args.batch_size, flag='False')
train_loader = mlt_dataprocess.Dataset_Generator(channel_data_train, Input_data_train, mask_train,
                                                 characteristic_index, target_index, args.batch_size, flag='True')

print('Length of Training set:', len(train_loader), 'Length of Test set:', len(test_loader))

# model initialization   初始化模型，模型进行装载

if args.state == 1:
    model = mlt_model.mltask(in_channels=len(characteristic_index), out_channels=64).to(args.device)
else:
    model = mlt_model.mltask(in_channels=len(characteristic_index), out_channels=64).to(args.device)  # model的shape是一个1

# train   对应不同状态下的环境，选择不同的的训练方式
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
