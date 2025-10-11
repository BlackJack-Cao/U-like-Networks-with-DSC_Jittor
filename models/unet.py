""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""


import jittor as jt
import jittor.nn as nn
import jittor.nn as F
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import config_setting as cfg
from models import DMSK,TTT

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
        position_ids = jt.arange(x_norm.shape[1]).unsqueeze(0).expand(x_norm.shape[0], -1)
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
        position_ids = jt.arange(x_norm.shape[1]).unsqueeze(0).expand(x_norm.shape[0], -1)
        x_TTT = self.TTTLinear(x_norm, position_ids=position_ids)
        out = x_TTT.reshape(B, n_tokens, *img_dims)

        return out

    def forward(self, x):
        if x.dtype == jt.float16:
            x = x.astype(jt.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
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
        diffY = jt.array([x2.shape[2] - x1.shape[2]])
        diffX = jt.array([x2.shape[3] - x1.shape[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = jt.concat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1_processed = self.dmsk_ttt1(x1)
        x2_processed = self.dmsk_ttt2(x2)
        x3_processed = self.dmsk_ttt3(x3)
        x4_processed = self.dmsk_ttt4(x4)

        x = self.up1(x5, x4_processed)
        x = self.up2(x, x3_processed)
        x = self.up3(x, x2_processed)
        x = self.up4(x, x1_processed)
        logits = self.outc(x)

        return  jt.sigmoid(logits)

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    dummy_input = jt.randn(1, 3, 256, 256)
    flops, params = profile(net, (dummy_input,))
    print('flops: %.2f M, params: %.2f k' % (flops / 1000000, params / 1000))
    print('net total parameters:', sum(param.numel() for param in net.parameters()))