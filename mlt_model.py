sigmoid = nn.Sigmoid()


# model 5 多任务学习模型
class mltask(nn.Module):
    def __init__(self, in_channels, out_channels):
        """

        :param in_channels: 输入数据通道数（特征维度）
        :param out_channels: 输出数据通道数（特征维度）
        """
        super(mltask, self).__init__()
        self.net1 = Attention(in_channels, out_channels)   # Attention模块
        self.net2 = net2(out_channels)                     # Fine_tune模块
        self.weights = nn.Parameter(torch.ones(6).float())

    def forward(self, x, args):
        """

        :param x: 输入原始数据
        :param args: 命令控制参数
        :return: 返回的传播结果
        """
        y = self.net1(x)
        if args.period == 1:
            y = self.net2(y)
        else:
            y = self.net2(y.detach())
        return y


class net1(nn.Module):
    def __init__(self, in_channels, out_channels):
        """

        :param in_channels: backbone部分网络输入数据维度
        :param out_channels: backbone部分网络输出数据维度
        """
        super(net1, self).__init__()
        # 定义卷积块
        self.Convv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),   # 卷积操作（卷积后channel尺度为64）
            nn.ReLU(),           # 激活函数
            nn.Conv2d(64, in_channels, kernel_size=(3, 3), padding=1),   # 卷积操作（卷积后channel尺度为in_channels）
            nn.ReLU()            # 激活函数
        )
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)   # 卷积层（卷积后为out_channels）

    def forward(self, x):
        x1 = self.Convv(x)
        x1 = x1 + x         # 残差连接
        x1 = self.Convv(x1)
        x1 = x1 + x         # 残差连接
        return self.final(x1)    # 输出


class Attention(nn.Module):                   # 含有注意力机制的backbone
    def __init__(self, in_channels, out_channels):
        """

        :param in_channels: backbone部分网络输入数据维度
        :param out_channels: backbone部分网络输出数据维度
        """
        super(Attention, self).__init__()
        self.Convv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),   # 卷积操作（卷积后channel尺度为64）
            nn.ReLU()           # 激活函数
        )
        self.Convv2 = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=(3, 3), padding=1),   # 卷积操作（卷积后channel尺度为in_channels）
            nn.ReLU()           # 激活函数
        )
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)   # 卷积层（卷积后为out_channels）
        self.attention = SEAttention(channel=64, reduction=7)     # attention模块，对权重进行调整

    def forward(self, x):
        x1 = self.Convv1(x)         # 卷积块 1
        x1 = self.attention(x1)     # attention模块，对权重进行调整
        x1 = self.Convv2(x1)        # 卷积块 2
        x1 = x1 + x                 # 残差连接
        x1 = self.Convv1(x1)        # 卷积块 1
        x1 = self.Convv2(x1)        # 卷积块 2
        x1 = x1 + x                 # 残差连接
        return self.final(x1)


class net2(nn.Module):
    def __init__(self, in_channels):
        super(net2, self).__init__()
        self.power = nn.Sequential(         # power（接收功率）的输出
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),   # 卷积操作（卷积后channel尺度为16）
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),             # 卷积操作（卷积后channel尺度为1）

        )
        self.thetaphi = nn.Sequential(      # thetaphi（均方根方位角和均方根俯仰角扩展）的输出
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),   # 卷积操作（卷积后channel尺度为64）
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=(3, 3), padding=1),            # 卷积操作（卷积后channel尺度为16）
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=(3, 3), padding=1)              # 卷积操作（卷积后channel尺度为2）
        )
        self.poweratio = nn.Sequential(     # poweratio（多径功率比）的输出
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),   # 卷积操作（卷积后channel尺度为16）
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),             # 卷积操作（卷积后channel尺度为1）

        )
        self.losn = nn.Sequential(          # losn（los/nlos）的输出
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),   # 卷积操作（卷积后channel尺度32）
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1),            # 卷积操作（卷积后channel尺度为16）
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv2d(16, 3, kernel_size=(3, 3), padding=1),             # 卷积操作（卷积后channel尺度为3）

        )
        self.delay = nn.Sequential(        # delay（均方根时延扩展）的输出
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),   # 卷积操作（卷积后channel尺度16）
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),             # 卷积操作（卷积后channel尺度为1）

        )
        self.sfmx = nn.Softmax(dim=1)

    def forward(self, x):
        thephi = self.thetaphi(x)  # out_channel = 2
        poweratio = self.poweratio(x)  # out_channel = 1
        power = self.power(x)  # out_channel = 1
        delay = self.delay(x)  # out_channel = 1
        los = - torch.log(self.sfmx(self.losn(x)))  # out_channel = 1
        return thephi, poweratio, power, delay, los
