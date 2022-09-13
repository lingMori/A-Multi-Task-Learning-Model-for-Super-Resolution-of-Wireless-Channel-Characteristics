from torchvision.models import resnet50
import torch
import torchvision.models as models
# import torch
import mlt_model
import mlt_dataprocess
import numpy as np
from thop import profile
from ptflops import get_model_complexity_info
from ptflops import get_model_complexity_info

characteristic_index = np.array([0, 1, 2, 3, 4, 5, 6])
# model = models.resnet50()   #调用官方的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoints = 'D://python_project//Final_mode_of_MTL//MTL_best_weight//MTL_best.pth'
model = mlt_model.mltask(in_channels=len(characteristic_index), out_channels=64).to(device)
# model = mlt_dataprocess.resume(model, checkpoints)

dummy_input = torch.rand(1, 7, 200, 200).to(device)
flops, params = get_model_complexity_info(model, (7, 200, 200), as_strings=True, print_per_layer_stat=True)
print('FLOPs:', flops, 'params', params)
