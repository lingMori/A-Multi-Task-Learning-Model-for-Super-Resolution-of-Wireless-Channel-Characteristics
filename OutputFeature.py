import mlt_model
import mlt_dataprocess
import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import scipy.io

# ===============================================================================================
features_in_hook = []
features_out_hook = []


def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None


def transform(tensor):
    # tensor = torch.Tensor.cpu()
    numpy = tensor.cpu().detach().numpy()
    print(numpy.shape)
    numpy /= 0.0078125
    numpy += 127.5
    numpy = numpy / 255
    print(numpy.shape)
    numpy = np.transpose(numpy, (1, 2, 0))
    print(numpy.shape)
    return numpy


# ===============================================================================================

parser = argparse.ArgumentParser(description='Mltask prediction & classification')
parser.add_argument('--period', type=int, default=1)
args = parser.parse_args()
scale = 2
index = 5
characteristic_index = np.array([0, 1, 2, 3, 4, 5, 6])

# index 是随机的一个区域 顺序是 K phi theta p t los

channel_data = np.load("data/BJ_1_2_train.npy")
channel_data_test = torch.from_numpy(channel_data).type(torch.float32)
channel_data_test = channel_data_test[[index], :, :, :]
# b = channel_data_test[0, 6, :, :].numpy()
model = mlt_model.mltask(in_channels=len(characteristic_index), out_channels=32)
print(model)
# ===============================================================================================
layer_name = 'net1.final'
for (name, module) in model.named_modules():
    print(name)
    if name == layer_name:
        module.register_forward_hook(hook=hook)

# ===============================================================================================

model = mlt_dataprocess.resume(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                                   'City_without_BJ_scale=2.pth'))
Origin = channel_data_test.squeeze().numpy()
LR_Data = F.interpolate(channel_data_test, scale_factor=1 / scale, mode='nearest').squeeze().numpy()

Output = mlt_dataprocess.INterplate(channel_data_test, scale=scale, MODE='bilinear')
with torch.no_grad():
    thephi, poweratio, power, delay, lostemp = model(Output, args)
los = torch.argmin(lostemp, dim=1, keepdim=True)
los = los - 1
a = los.numpy()

HR_Data = torch.cat((thephi, poweratio, power, delay, los), dim=1).squeeze().numpy()
feature = features_out_hook[0].cpu().detach().squeeze().numpy()

scipy.io.savemat('result_test.mat',
                 mdict={'HR_Data': HR_Data, 'LR_Data': LR_Data, 'Origin': Origin, 'feature': feature})
