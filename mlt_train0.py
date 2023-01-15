import sys
import time
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import os
import torch
import mlt_dataprocess
import mlt_loss
from pathlib import Path


def train(model, args, train_loader, test_loader):
    trloss = np.repeat(np.nan, 6, axis=0).tolist()   # train_loss空向量的创建
    teloss = np.repeat(np.nan, 7, axis=0).tolist()   # test_loss 空向量的创建

    traloss = mlt_loss.traloss0()
    tesloss = mlt_loss.tesloss0()
    uncertainty = mlt_loss.bploss(len(args.target_index), args).to(args.device)
    # optimizer
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': uncertainty.parameters(), 'lr': args.lr}
    ])
    # scheduler
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    since = time.time()
    sys.stdout = mlt_dataprocess.Record(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', args.logname))
    csvfile = Path(os.path.join('result', args.csvname))
    if csvfile.is_file():
        os.remove(csvfile)
    list = ['epoch', 'phi', 'theta', 'K', 'P', 'T', 'TPR', 'FPR']
    data = pd.DataFrame([list])
    data.to_csv(os.path.join('result', args.csvname), mode='a', header=None, index=False)
    for epoch in range(args.epochs):
        epoch_loss = torch.Tensor([0]).to(args.device)
        for Data, Label, mask in (train_loader):
            # if epoch == 0:
            #     break
            data = Data.to(args.device)
            label = Label.to(args.device)
            mask = mask.to(args.device)
            output = model(data, args)
            loss = traloss(output, label, mask)  # c

            bploss = uncertainty(loss)      # 这个bploss到底是干啥用的？
            # bploss = torch.sum(loss)
            optimizer.zero_grad()
            bploss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss / len(train_loader)
        if epoch % 5 == 0: print(epoch_loss)

        with torch.no_grad():

            epoch_test_loss = torch.Tensor([0]).to(args.device)

            for Data, Label, mask in test_loader:
                data = Data.to(args.device)
                label = Label.to(args.device)
                mask = mask.to(args.device)

                testoutput = model(data, args)
                loss = tesloss(testoutput, label, mask)
                epoch_test_loss = epoch_test_loss + loss / len(test_loader)
        j = 0
        for i in range(5):
            if len([k for k in range(len(args.target_index)) if args.target_index[k] == i + 1]) > 0:
                trloss[i] = epoch_loss[j].item()
                teloss[i] = epoch_test_loss[j].item()
                j = j + 1
        # print(j)

        time_elapsed = time.time() - since
        print("====================================================")
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        teloss = np.array(teloss)
        teloss[0:2] = teloss[0:2] * 180
        teloss[2:5] = teloss[2:5] * 100
        teloss = teloss.tolist()
        list = [epoch + 1] + teloss
        data = pd.DataFrame([list])
        data.to_csv(os.path.join('result', args.csvname), mode='a', header=False, index=False)

        print(uncertainty.coefficient.detach().squeeze().cpu().numpy())

        # print(parameters for parameters in uncertainty.parameters())

        print(f"Epoch : {epoch + 1} (trainloss) phi :{trloss[0]:.4f}   theta :{trloss[1]:.4f}   K :{trloss[2]:.4f}\
   P :{trloss[3]:.4f}   T :{trloss[4]:.4f}   L :{trloss[5]:.4f}")
        print(f"Epoch : {epoch + 1} (testloss)  phi :{teloss[0]:.4f}   theta :{teloss[1]:.4f}   K :{teloss[2]:.4f}\
   P :{teloss[3]:.4f}   T :{teloss[4]:.4f}   TPR :{teloss[5]:.4f}   FPR :{teloss[6]:.4f}\n")
