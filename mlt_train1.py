import sys
import time
import numpy as np
import pandas as pd
import torch.optim as optim
import os
import torch
from pathlib import Path
import mlt_dataprocess
import mlt_loss


def train(model, args, train_loader, test_loader):
    # define
    """

    :param model: 传入多任务学习的超分辨率模型
    :param args: 定义的宏参数
    :param train_loader: 训练集加载器
    :param test_loader: 测试集加载器
    :return: null
    """
    work_dir = os.path.dirname(os.path.abspath(__file__))   # 获取当前文件目录
    Result = os.path.join(work_dir,'result')       # 将数据路径添加入主路径当中（适配windows和Linux）
    """loss function & STDE function define"""
    train_loss = np.repeat(np.nan, 6, axis=0).tolist()          # train loss空列表创建
    test_loss = np.repeat(np.nan, 7, axis=0).tolist()           # test loss空列表创建
    test_std_error = np.repeat(np.nan, 5, axis=0).tolist()      # train std空列表创建
    train_std_error = np.repeat(np.nan, 5, axis=0).tolist()     # test std空列表创建

    train_lossfunction1 = mlt_loss.traloss0()           # 导入train loss损失函数的传播模型（非Los/NLos）
    train_lossfunction2 = mlt_loss.traloss2()           # 导入train loss损失函数的传播模型（Los/NLos）
    test_lossfunction1 = mlt_loss.tesloss0()            # 导入test loss损失函数的传播模型（非Los/NLos）
    test_lossfunction2 = mlt_loss.tesloss2()            # 导入test loss损失函数的传播模型（Los/NLos）
    stdfunction = mlt_loss.Std()                   # 定义STDE标准差极端传播模型（无Los/NLos）
    Uncertainty = mlt_loss.bploss(len(args.target_index), args).to(args.device)

    # scheduler
    # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    """training profiles define"""
    # training
    since = time.time()          # 记录训练开始时间
    sys.stdout = mlt_dataprocess.Record(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', args.logname))
    csvfile = Path(os.path.join('result', args.csvname))
    if csvfile.is_file():
        os.remove(csvfile)

    # 顺序是 K phi theta p t los
    list = ['epoch', 'K_mae', 'Phi_mae', 'Theta_mae', 'P_mae', 'T_mae', 'TPR_mae', 'FPR_mae', 'K_std', 'Phi_std',
            'Theta_std', 'P_std', 'T_std']
    data = pd.DataFrame([list])     # 创建DataFrame数据结构框架
    data.to_csv(os.path.join(Result, args.csvname), mode='a', header=None, index=False)     # 写入csv文件

    print("*************************第一阶段:********************************* ")
    for epoch in range(args.epochs_P1):
        P1_epoch_trainloss = torch.Tensor([0]).to(args.device)          # 创建train loss的tensor张量
        P1_epoch_teststderror = torch.Tensor([0]).to(args.device)       # 创建test std的tensor张量
        P1_epoch_testloss = torch.Tensor([0]).to(args.device)           # 创建test loss的tensor张量
        # optimizer  定义优化器
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': Uncertainty.parameters(), 'lr': args.lr}])
        """训练阶段，从train——loader中加载需要的数据"""
        for Data, Label, mask in (train_loader):
            data = Data.to(args.device)             # 经过处理的数据
            label = Label.to(args.device)           # 高分辨率数据
            mask = mask.to(args.device)             # 蒙板

            thephi, poweratio, power, delay, los = model(data, args)       # 获得模型的输出

            loss1thephi = train_lossfunction1(thephi, label[:, 0:2, :, :], mask)                    # thetaphi（均方根方位角和均方根俯仰角扩展）的loss
            loss2poweratio = train_lossfunction1(poweratio, label[:, 2:3, :, :], mask)              # poweratio（多径功率比）的loss
            loss3power = train_lossfunction1(power, label[:, 3:4, :, :], mask)                      # power（接收功率）的loss
            loss4delay = train_lossfunction1(delay, label[:, 4:5, :, :], mask)                      # delay（均方根时延扩展）的loss
            lossLOS = train_lossfunction2(los, label[:, [-1], :, :], mask).unsqueeze(0)             # losn（los/nlos）的loss

            loss = torch.cat((loss1thephi, loss2poweratio, loss3power, loss4delay, lossLOS), dim=0) # loss合并

            bploss = Uncertainty(loss)              # Uncertainty weight对不同任务loss权重进行均衡
            optimizer.zero_grad()
            bploss.backward()                       # 反向传播
            optimizer.step()
            P1_epoch_trainloss = P1_epoch_trainloss + loss / len(train_loader)       # 将训练结果保存

        # print(model.state_dict()['net1.Convv.0.bias'])

        with torch.no_grad():
            TPRmean = torch.Tensor([0]).to(args.device)
            FPRmean = torch.Tensor([0]).to(args.device)
            for Data, Label, mask in test_loader:
                data = Data.to(args.device)         # 经过处理的数据
                label = Label.to(args.device)       # 高分辨率数据
                mask = mask.to(args.device)         # 蒙板

                thephi, poweratio, power, delay, los = model(data, args)     # 获得模型的输出

                """loss function output"""
                loss1thephi = test_lossfunction1(thephi, label[:, 0:2, :, :], mask)             # thetaphi（均方根方位角和均方根俯仰角扩展）的loss
                loss2poweratio = test_lossfunction1(poweratio, label[:, 2:3, :, :], mask)       # poweratio（多径功率比）的loss
                loss3power = test_lossfunction1(power, label[:, 3:4, :, :], mask)               # power（接收功率）的loss
                loss4delay = test_lossfunction1(delay, label[:, 4:5, :, :], mask)               # delay（均方根时延扩展）的loss
                TPR, FPR = test_lossfunction2(los, label[:, [-1], :, :], mask)                  # losn（los/nlos）的loss
                loss = torch.cat((loss1thephi, loss2poweratio, loss3power, loss4delay), dim=0)      # loss合并
                """std error plus"""
                std1thephi = stdfunction(thephi, label[:, 0:2, :, :], mask)             # thetaphi（均方根方位角和均方根俯仰角扩展）的std
                std2poweratio = stdfunction(poweratio, label[:, 2:3, :, :], mask)       # poweratio（多径功率比）的loss
                std3power = stdfunction(power, label[:, 3:4, :, :], mask)               # power（接收功率）的loss
                std4delay = stdfunction(delay, label[:, 4:5, :, :], mask)               # delay（均方根时延扩展）的loss
                std = torch.cat((std1thephi, std2poweratio, std3power, std4delay), dim=0)       # std合并

                P1_epoch_testloss = P1_epoch_testloss + loss / len(test_loader)             # loss训练结果保存
                P1_epoch_teststderror = P1_epoch_teststderror + std / len(test_loader)      # std训练结果保存

                TPRmean += TPR / len(test_loader)
                FPRmean += FPR / len(test_loader)

        # result recording
        counter = 0
        for index_I in range(5):
            if len([k for k in range(len(args.target_index)) if args.target_index[k] == index_I + 1]) > 0:
                train_loss[index_I] = P1_epoch_trainloss[counter].item()
                test_loss[index_I] = P1_epoch_testloss[counter].item()
                test_std_error[index_I] = P1_epoch_teststderror[counter].item()
                counter = counter + 1
        train_loss[5] = P1_epoch_trainloss[5].item()
        test_loss[5] = TPRmean.item()
        test_loss[6] = FPRmean.item()

        time_elapsed = time.time() - since
        print("====================================================")
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        test_loss = (np.array(test_loss) * [100, 180, 180, 100, 100, 100, 100]).tolist()    # 训练结果标准化
        train_loss = (np.array(train_loss) * [100, 180, 180, 100, 100, 1]).tolist()         # 训练结果标准化
        test_std_error = (np.array(test_std_error) * [100, 180, 180, 100, 100]).tolist()    # 训练结果标准化

        list = [epoch + 1] + test_loss
        list = list + test_std_error
        data = pd.DataFrame([list])
        data.to_csv(os.path.join(Result, args.csvname), mode='a', header=None, index=False)     # 训练结果写入csv文件
        """打印所有的训练结果"""
        print(Uncertainty.coefficient.detach().squeeze().cpu().numpy())
        print(
            f"Epoch : {epoch + 1} (trainloss) K :{train_loss[0]:.4f}   phi :{train_loss[1]:.4f}   theta :{train_loss[2]:.4f}\
   P :{train_loss[3]:.4f}   T :{train_loss[4]:.4f}   L :{train_loss[5]:.4f}")
        print(
            f"Epoch : {epoch + 1} (testloss)  K :{test_loss[0]:.4f}   phi :{test_loss[1]:.4f}   theta :{test_loss[2]:.4f}\
   P :{test_loss[3]:.4f}   T :{test_loss[4]:.4f}   TPR :{test_loss[5]:.4f}   FPR :{test_loss[6]:.4f}")
        print(
            f"Epoch : {epoch + 1} (test_std_error)  K :{test_std_error[0]:.4f}   phi :{test_std_error[1]:.4f}   theta :{test_std_error[2]:.4f}\
   P :{test_std_error[3]:.4f}   T :{test_std_error[4]:.4f}\n")

    print("*************************第二阶段:********************************* ")
    args.period = 2
    for epoch in range(args.epochs_P2):
        P2_epoch_trainloss = torch.Tensor([0]).to(args.device)          # 创建train loss的tensor张量
        P2_epoch_teststderror = torch.Tensor([0]).to(args.device)       # 创建test std的tensor张量
        P2_epoch_testloss = torch.Tensor([0]).to(args.device)           # 创建test loss的tensor张量
        P2_epoch_trainstderror = torch.Tensor([0]).to(args.device)      # 创建train std的tensor张量
        # optimizer
        optimizer = optim.Adam([
            {'params': model.net2.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}])
        for Data, Label, mask in (train_loader):
            data = Data.to(args.device)             # 经过处理的数据
            label = Label.to(args.device)           # 高分辨率数据
            mask = mask.to(args.device)             # 蒙板
            thephi, poweratio, power, delay, los = model(data, args)        # 获得模型的输出

            """loss function output"""
            loss1thephi = train_lossfunction1(thephi, label[:, 0:2, :, :], mask)                # thetaphi（均方根方位角和均方根俯仰角扩展）的loss
            loss2poweratio = train_lossfunction1(poweratio, label[:, 2:3, :, :], mask)          # poweratio（多径功率比）的loss
            loss3power = train_lossfunction1(power, label[:, 3:4, :, :], mask)                  # power（接收功率）的loss
            loss4delay = train_lossfunction1(delay, label[:, 4:5, :, :], mask)                  # delay（均方根时延扩展）的loss
            lossLOS = train_lossfunction2(los, label[:, [-1], :, :], mask).unsqueeze(0)  # 1    # losn（los/nlos）的loss
            loss = torch.cat((loss1thephi, loss2poweratio, loss3power, loss4delay, lossLOS), dim=0) # loss合并

            trstd1thephi = stdfunction(thephi, label[:, 0:2, :, :], mask)               # thetaphi（均方根方位角和均方根俯仰角扩展）的std
            trstd2poweratio = stdfunction(poweratio, label[:, 2:3, :, :], mask)         # poweratio（多径功率比）的loss
            trstd3power = stdfunction(power, label[:, 3:4, :, :], mask)                 # power（接收功率）的loss
            trstd4delay = stdfunction(delay, label[:, 4:5, :, :], mask)                 # delay（均方根时延扩展）的loss
            std = torch.cat((trstd1thephi, trstd2poweratio, trstd3power, trstd4delay), dim=0)           # std合并

            optimizer.zero_grad()
            """
             接下来就是对所有loss函数分别进行反向传播
             """

            # torch.sum(loss).backward()
            torch.sum(loss1thephi).backward()
            torch.sum(loss2poweratio).backward()
            torch.sum(loss3power).backward()
            torch.sum(loss4delay).backward()
            torch.sum(lossLOS).backward()
            optimizer.step()

            P2_epoch_trainstderror = P2_epoch_trainstderror + std / len(train_loader)           # std训练结果保存
            P2_epoch_trainloss = P2_epoch_trainloss + loss / len(train_loader)                  # loss训练结果保存

        # print(model.state_dict()['net2.power.0.bias'])

        with torch.no_grad():
            TPRmean = torch.Tensor([0]).to(args.device)
            FPRmean = torch.Tensor([0]).to(args.device)
            for Data, Label, mask in test_loader:
                data = Data.to(args.device)                 # 经过处理的数据
                label = Label.to(args.device)               # 高分辨率数据
                mask = mask.to(args.device)                 # 蒙板
                thephi, poweratio, power, delay, los = model(data, args)

                """loss function output"""
                loss1thephi = test_lossfunction1(thephi, label[:, 0:2, :, :], mask)             # thetaphi（均方根方位角和均方根俯仰角扩展）的loss
                loss2poweratio = test_lossfunction1(poweratio, label[:, 2:3, :, :], mask)       # poweratio（多径功率比）的loss
                loss3power = test_lossfunction1(power, label[:, 3:4, :, :], mask)               # power（接收功率）的loss
                loss4delay = test_lossfunction1(delay, label[:, 4:5, :, :], mask)               # delay（均方根时延扩展）的loss
                TPR, FPR = test_lossfunction2(los, label[:, [-1], :, :], mask)                  # losn（los/nlos）的loss
                loss = torch.cat((loss1thephi, loss2poweratio, loss3power, loss4delay), dim=0)  # loss合并

                """std error plus"""
                std1thephi = stdfunction(thephi, label[:, 0:2, :, :], mask)                 # thetaphi（均方根方位角和均方根俯仰角扩展）的std
                std2poweratio = stdfunction(poweratio, label[:, 2:3, :, :], mask)           # poweratio（多径功率比）的loss
                std3power = stdfunction(power, label[:, 3:4, :, :], mask)                   # power（接收功率）的loss
                std4delay = stdfunction(delay, label[:, 4:5, :, :], mask)                   # delay（均方根时延扩展）的loss
                std = torch.cat((std1thephi, std2poweratio, std3power, std4delay), dim=0)   # std合并

                P2_epoch_testloss = P2_epoch_testloss + loss / len(test_loader)
                P2_epoch_teststderror = P2_epoch_teststderror + std / len(test_loader)
                TPRmean += TPR / len(test_loader)
                FPRmean += FPR / len(test_loader)


        """result recording"""
        counter = 0
        for index_I in range(5):
            if len([k for k in range(len(args.target_index)) if args.target_index[k] == index_I + 1]) > 0:
                train_loss[index_I] = P2_epoch_trainloss[counter].item()
                test_loss[index_I] = P2_epoch_testloss[counter].item()
                test_std_error[index_I] = P2_epoch_teststderror[counter].item()
                train_std_error[index_I] = P2_epoch_trainstderror[counter].item()
                counter = counter + 1
        train_loss[5] = P2_epoch_trainloss[5].item()
        test_loss[5] = TPRmean.item()
        test_loss[6] = FPRmean.item()

        time_elapsed = time.time() - since
        print("====================================================")
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        test_loss = (np.array(test_loss) * [100, 180, 180, 100, 100, 100, 100]).tolist()        # 训练结果标准化
        train_loss = (np.array(train_loss) * [100, 180, 180, 100, 100, 1]).tolist()             # 训练结果标准化
        test_std_error = (np.array(test_std_error) * [100, 180, 180, 100, 100]).tolist()        # 训练结果标准化
        train_std_error = (np.array(train_std_error) * [100, 180, 180, 100, 100]).tolist()      # 训练结果标准化
        list = [epoch + 1] + test_loss
        list = list + test_std_error
        data = pd.DataFrame([list])
        data.to_csv(os.path.join(Result, args.csvname), mode='a', header=None, index=False)

        print(
            f"Epoch : {epoch + 1} (trainloss) K :{train_loss[0]:.4f}   phi :{train_loss[1]:.4f}   theta :{train_loss[2]:.4f}\
    P :{train_loss[3]:.4f}   T :{train_loss[4]:.4f}   L :{train_loss[5]:.4f}")
        print(
            f"Epoch : {epoch + 1} (train_std_error) K :{train_std_error[0]:.4f}   phi :{train_std_error[1]:.4f}   theta :{train_std_error[2]:.4f}\
    P :{train_std_error[3]:.4f}   T :{train_std_error[4]:.4f}")
        print(
            f"Epoch : {epoch + 1} (testloss)  K :{test_loss[0]:.4f}   phi :{test_loss[1]:.4f}   theta :{test_loss[2]:.4f}\
    P :{test_loss[3]:.4f}   T :{test_loss[4]:.4f}   TPR :{test_loss[5]:.4f}   FPR :{test_loss[6]:.4f}")
        print(
            f"Epoch : {epoch + 1} (test_std_error)  K :{test_std_error[0]:.4f}   phi :{test_std_error[1]:.4f}   theta :{test_std_error[2]:.4f}\
    P :{test_std_error[3]:.4f}   T :{test_std_error[4]:.4f}\n")

        # save model parameters and weights
        if epoch == args.epochs_P2 - 5:
            last_model_wts = model.state_dict()
            save_path_last = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', args.wight_lastname)
            torch.save(last_model_wts, save_path_last)
