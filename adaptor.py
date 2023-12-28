import torch.nn as nn
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class FGenerator(nn.Module):
    def __init__(self,input_channel=1):
        super(FGenerator, self).__init__()
        
        self.initial_conv = nn.Conv3d(input_channel, 64, kernel_size=3, padding=1)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(64)
        )
        self.out_conv = nn.Conv3d(64, input_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.out_conv(x)
        return x
class FourierTransform(nn.Module):
    def __init__(self,input_channel=2):
        super(FourierTransform, self).__init__()
        # 初始化实部和虚部的可训练参数
        self.generator = FGenerator(input_channel)
    def forward(self, x):
        # 执行傅里叶变换
        x = x.float()
        fft_result = torch.fft.fftn(x)

        # 提取实部和虚部
        real_part = fft_result.real
        imag_part = fft_result.imag

        out = self.generator(torch.cat([real_part,imag_part],dim=1))
        modified_real = out[:,0:1]
        modified_imag = out[:,1:2]
        # 重新组合为复数张量
        modified_fft_result = torch.complex(modified_real, modified_imag)

        # 执行逆傅里叶变换
        ifft_result = torch.fft.ifftn(modified_fft_result).real

        return ifft_result.half()

class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        
        self.initial_conv = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(64)
        )

        self.out_conv = nn.Conv3d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.out_conv(x)
        return torch.sigmoid(x)