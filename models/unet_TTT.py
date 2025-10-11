import jittor as jt
import gc
import matplotlib.pyplot as plt
import numpy as np

gc.collect()
import jittor.nn as nn
import jittor.nn as F
import os
import sys
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import config_setting as cfg
from models import DMSK,TTT
from torch.cuda.amp import autocast
from models.feature_visualizer import SkinFeatureVisualizer
class TTTLayer(nn.Module):
    def __init__(self, dim, layer_idx, d_state = 16, channel_token = False):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # self.mamba = Mamba(
        #         d_model=dim, # Model dimension d_model
        #         d_state=d_state,  # SSM state expansion factor
        #         d_conv=d_conv,    # Local convolution width
        #         expand=expand,    # Block expansion factor
        # )
        self.TTTConfig = TTT.TTTConfig(hidden_size=dim, num_attention_heads=8, num_hidden_layers=d_state)
        self.TTTLinear = TTT.TTTLinear(self.TTTConfig, layer_idx)
        self.channel_token = channel_token ## whether to use channel as tokens

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        position_ids = torch.arange(x_norm.size(1), device=x_norm.device).unsqueeze(0).expand(x_norm.size(0), -1)
        x_TTT = self.TTTLinear(x_norm, position_ids=position_ids)
        out = x_TTT.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        position_ids = torch.arange(x_norm.size(1), device=x_norm.device).unsqueeze(0).expand(x_norm.size(0), -1)
        x_TTT = self.TTTLinear(x_norm, position_ids=position_ids)
        out = x_TTT.reshape(B, n_tokens, *img_dims)

        return out

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    #定义类的初始化方法，在创建DoubleConv对象时自动调用
    def __init__(self, in_channels, out_channels):
        #in_channels和out_channels是输入通道和输出通道数，决定输入和输出的特征图数量
        super().__init__()
        #定义一个顺序容器，将多个操作按顺序串联起来
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #padding=1，保证输入和输出特征图的空间尺寸相同
            nn.BatchNorm2d(out_channels),
            #输入的通道数是out_channels，对每一通道的数据进行归一化，减少特征分布变化，加快训练速度
            nn.ReLU(inplace=True),
            #inplace=True表示直接在输入数据上操作，不创建新的数据，节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    #x是输入的张量，让它执行两次“卷积 -> 批归一化 -> 激活”的操作，并输出处理后的特征图


#用于图像的下采样，通过 最大池化（MaxPooling） 操作对输入的特征图进行下采样，使宽高减半，
#然后通过之前定义的 DoubleConv 模块提取特征
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

#定义类的初始化方法，创建 Down 对象时会自动调用
#假设输入张量x形状为(batch_size,in_channels,height,width)
#经过最大池化层后输出形状为：(batch_size,in_channels//2,height,width//2)
#经过DoubleConv输出形状为：(batch_size,out_channels//2,height,width//2)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #定义一个顺序容器（nn.Sequential），将最大池化和双卷积操作按顺序组织起来。
        self.maxpool_conv = nn.Sequential(
            #定义二维最大池化层，池化窗口的大小是2x2
            #让宽度和高度减半
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
#常用于神经网络的输出层。它的作用是通过一个1x1的卷积将特征图的通道数调整为目标输出通道数。
#这个模块的输入是网络的特征图，而输出可以是例如分类结果、分割概率图等，具体取决于网络的任务。
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        #调用父类 nn.Module 的初始化方法，确保继承的功能正确初始化。
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        #1×1 的卷积不会改变特征图的宽和高

    def forward(self, x):
        return self.conv(x)

class UNet_TTT(nn.Module):
    #activefunc：激活函数的类型（可选 'relu'、'gelu'、'tanh'）
    #droprate：Dropout 的概率，用于防止过拟合。
    #kernel_size：卷积核大小
    #n_channels：输入图像的通道数（如 RGB 图像通道数为 3）。
    #n_classes：输出的类别数量（分割任务的类别数）。
    #bilinear：是否使用双线性插值进行上采样。
    def __init__(self, 
                 activefunc,
                 droprate,
                 kernel_size,
                 n_channels, 
                 n_classes, 
                 bilinear=True,
                 save_dir='/workspace/UNet-TTT/data',
                 ):
        super(UNet_TTT, self).__init__()
        #根据传入的参数activatefunc的值，选择使用下面三个作为激活函数
        if activefunc == 'relu':
            self.act = nn.ReLU()
        elif activefunc == 'gelu':
            self.act = nn.GELU()
        elif activefunc == 'tanh':
            self.act = nn.Tanh()

        self.visualizer = SkinFeatureVisualizer(save_dir=save_dir)
        #定义一个 Dropout 层，用于随机丢弃神经元，防止过拟合。
        self.drop = nn.Dropout(p=droprate)  #
        self.ker = kernel_size  #记录卷积核大小
        #计算卷积的填充（padding）大小，使卷积后的特征图保持合理尺寸。
        self.pad = (self.ker - 1) // 2  

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

    #DoubleConv有两个参数，一个in_channels，一个out_channels，将输入通道数转为输出通道数
    #根据Unet的网络架构图，输入首先进来后要进行卷积操作，将通道数变成64，再进行四次下采样
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.dmsk_ttt1 = nn.Sequential(DMSK.DMSKModule(64), TTTLayer(64, layer_idx=0))
        self.dmsk_ttt2 = nn.Sequential(DMSK.DMSKModule(128), TTTLayer(128, layer_idx=1))
        self.dmsk_ttt3 = nn.Sequential(DMSK.DMSKModule(256), TTTLayer(256, layer_idx=2))
        self.dmsk_ttt4 = nn.Sequential(DMSK.DMSKModule(512), TTTLayer(512, layer_idx=3))

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
    
        # top = x.clone()
        # hidden = self.saliency(top)
        # hidden = self.bottleneck(hidden)
        x1 = self.inc(x)
        # self.visualizer.save_skin_features(x1, '第一层处理前.png', x)
        x2 = self.down1(x1)
        # self.visualizer.save_skin_features(x2, '第二层处理前.png', x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #scale = 16：特征图放大比例
        #输出 y5 经过激活函数和 Dropout。

        x1_processed = self.dmsk_ttt1(x1)
        # self.visualizer.save_skin_features(x1_processed, '第一层处理后.png', x)
        x2_processed = self.dmsk_ttt2(x2)
        # self.visualizer.save_skin_features(x2_processed, '第二层处理后.png', x)
        x3_processed = self.dmsk_ttt3(x3)
        x4_processed = self.dmsk_ttt4(x4)

        y4 = self.act(self.up1(x5, x4_processed))
        y4 = self.drop(y4)

        y3 = self.act(self.up2(y4, x3_processed))
        y3 = self.drop(y3)

        y2 = self.act(self.up3(y3, x2_processed))
        y2 = self.drop(y2)

        y1 = self.act(self.up4(y2, x1_processed))
        y1 = self.drop(y1)
        y = self.outc(y1)

        #使用 torch.sigmoid 将输出值归一化到[0,1]便于后续处理（如二分类任务中的阈值化）
        return  torch.sigmoid(y)

if __name__ == '__main__':
    net = UNet_TTT(activefunc='relu',droprate=0.1,kernel_size=3,n_channels=3, n_classes=1).cuda()
    from thop import profile

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    skin_img = Image.open('/workspace/UNet-TTT/data/isic2017/val/images/ISIC_0003582.jpg').convert('RGB')
    skin_tensor = transform(skin_img).unsqueeze(0).cuda()
    output = net(skin_tensor)
    # dummy_input = torch.randn(1, 3, 256, 256).cuda()
    # flops, params = profile(net, (dummy_input,))
    # print('flops: %.2f M, params: %.2f k' % (flops / 1000000, params / 1000))
    # print('net total parameters:', sum(param.numel() for param in net.parameters()))
