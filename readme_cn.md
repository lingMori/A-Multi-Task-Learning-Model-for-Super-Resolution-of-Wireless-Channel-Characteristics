# 无线信道特性超分辨率的多任务学习模型

![banner](./image/net1.png)

![badge](./image/net2.png)



[![standard-readme compliant](https://img.shields.io/badge/Multi_Task%20-Super Resolution-brightgreen.svg?style=flat-square)](https://github.com/lingMori/A-Multi-Task-Learning-Model-for-Super-Resolution-of-Wireless-Channel-Characteristics)



我们提出了一种新颖的超分辨率（SR）模型，用于生成通道特性数据。该模型基于具有残差连接的多任务学习（MTL）卷积神经网络（CNN）。

实验表明，所提出的SR模型在均值绝对误差（MSE）和标准差（STDE）方面均能取得优异的性能。

## 目录

- [背景](#背景)
- [下载](#下载)
- [使用方法](#使用方法)
- [数据集](#数据集)
- [贡献者](#贡献者)
- [所属](#所属)

## 背景

### 研究现状

信道建模一直是通信系统设计和开发的核心部分，尤其是在5G和6G时代。随机信道建模和基于光线追踪 （RT） 的信道建模等传统方法在很大程度上依赖于测量数据或仿真，而这些数据或仿真通常既昂贵又耗时。

> 对于SCM，需要通道特性，如路径损耗（PL），传播条件（Los/NLos），延迟扩展，角度扩散和Rician K因子来生成通道系数，然后对无线通道进行建模。在不同环境中进行大规模信道测量，通常既耗时又昂贵，必须进行，以获得必要的信道特性数据。

> 另一方面，如果给定精确的传播环境和配置，基于RT的建模方法可以生成准确的通道数据，但代价是计算复杂性高，计算时间长。因此，一种快速可靠的信道特性数据生成方法将有效地解决这些局限性。

### 我们的贡献

我们并没有选择预测或估计的方法，而是为通道特征提出超分辨率（SR）模型。该模型基于残差连接的MTL CNN。我们的工作是：给定城市区域的三维模型和相应的EM参数，通过CloudRT平台输出通道特征数据集，并将该数据集用于SR模型训练。 数据集中的数据将通过处理成为低分辨率数据作为输入，原始的高分辨率数据作为真实值，MTL损耗用于更好地平衡多任务。我们通过消融研究和与其他DL模型的比较来评估我们提出的SR模型。

具体而言，我们做出了**以下贡献**：

1. 在密集城市地区，利用自主研发的**CloudRT**仿真平台进行RT仿真，通过构建三维电子地图进行射线追踪的仿真，并根据仿真结果构建信道特征数据集。

2. 提出一种基于残余网络的**MTL 的SR模型**。在损失函数中加入mask作为调整。Homoscedastic uncertainty用于平衡训练过程中的单一任务损失。在 CNN 模块中加入残差连接和上下采样技术，更好地获得 LR-HR 对之间的相互依赖关系，在网络中对特征进行采样，重建，然后利用重建结果的错误对采样结果进行改善。以获得更好的 SR 性能。

3. 提出了独立的训练和结果。所提出的 SR 方法通常比其他先进的深度学习模型表现得更好。与**基线**相比，所提出的方法可以在所有信道特性目标中实现非常好的SR结果，并且在较大的尺度因子下**明显低于**基线。所有结果都在`result`文件夹中。

   

## 下载

这个项目的部署基于 [Anaconda](https://www.anaconda.com/) 和 [PyTorch](https://pytorch.org/) 。如果您没有在本地安装它们，请按照官方链接进行安装。

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

所有模型和演示都基于PyTorch（1.12.1）和CUDA（11.3/10.2）。请在运行此项目之前检查您的环境配置版本是否正确。**本项目在Windows和Linux操作系统上都可以直接部署**。



## 使用方法

用户可以下载上面的代码，我们特别推荐下载zip来开发项目。

可以使用以下命令语句运行项目。

```
python mlt_main.py
```



## 数据集

工作人员可以使用  [CloudRT](http://cn.raytracer.cloud:9090/)  构建自己的渠道特征数据集，并将其打包到“.mat”文件中，然后将其放入“data”文件夹中即可训练自己的数据集。

**注意**: 如果您想获得**构建并检查无误的数据集**，请通过电子邮件与我们联系。我们的联系电子邮件：wangxiping@bjtu.edu.cn



## 贡献者

项目免费开源，使用者可以按照喜好和个人的方向进行开发！如有问题可以电子邮件联系：wangxiping@bjtu.edu.cn也可以在项目中的PR和Issue进行提问，我们收到信息后会尽快回复。

**注意事项**: 如果编辑自述文件，请遵守 [standard-readme](https://github.com/RichardLitt/standard-readme) 规范。

### Contributors

感谢所有为本项目做出贡献的开发者们！

![](./image/contributors.png)

## 所属

no license now.