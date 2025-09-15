import random
from torch import nn
import torch
from thop import profile


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_radio=16):
        super().__init__()
        self.channels = channels
        self.inter_channels = self.channels  // reduction_radio
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(  # 使用1x1卷积代替线性层，可以不用调整tensor的形状
            nn.Conv2d(self.channels, self.inter_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.inter_channels, self.channels,
                    kernel_size=1, stride=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        maxout = self.maxpool(x) # (b, c, 1, 1)
        avgout = self.avgpool(x) # (b, c, 1, 1)

        maxout = self.mlp(maxout) # (b, c, 1, 1)
        avgout = self.mlp(avgout) # (b, c, 1, 1)

        attention = self.sigmoid(maxout + avgout)  # (b, c, 1, 1)
        return attention


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(in_channels=2, out_channels=self.dim,
                              kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):  # (b, c, h, w)
        maxpool = x.argmax(dim=1, keepdim=True)  # (b, 1, h, w)
        avgpool = x.mean(dim=1, keepdim=True)  # (b, 1, h, w)
        out = torch.cat([maxpool, avgpool], dim=1)  # (b, 2, h, w)
        out = self.conv(out)  # (b, 1, h, w)
        attention = self.sigmoid(out)  # (b, 1, h, w)
        x  = attention * x
        return x


class ChannelShuffle(nn.Module):
    def __init__(self, channels, groups=16):
        super().__init__()
        assert channels % groups == 0
        self.channels = channels
        self.groups = groups
        self.channel_per_group = self.channels // self.groups

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        batch, _, series, modal = x.size()
        x = x.reshape(batch, self.groups, self.channel_per_group, series, modal)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch, self.channels, series, modal)
        return x


class DCEU(nn.Module):
    def __init__(self, in_dim, dilation):
        super(DCEU, self).__init__()
        self.in_dim = in_dim
        self.dim = self.in_dim // 2
        pad = int((3 - 1) / 2)
        self.dilation = dilation

        self.x_left = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, (3, 1), stride=1, padding=(pad+(self.dilation-1), 0), groups=self.dim, dilation=self.dilation),
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1, padding=0, groups=1),

            nn.Conv2d(self.dim, self.dim, (1, 3), stride=1, padding=(0, pad+(self.dilation-1)), groups=self.dim, dilation=self.dilation),
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1, padding=0, groups=1),
        )

        self.x_right = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, (1, 3), stride=1, padding=(0, pad + (self.dilation - 1)), groups=self.dim,
                      dilation=self.dilation),
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1, padding=0, groups=1),

            nn.Conv2d(self.dim, self.dim, (3, 1), stride=1, padding=(pad + (self.dilation - 1), 0), groups=self.dim,
                      dilation=self.dilation),
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1, padding=0, groups=1),
        )

        self.SRU = SpatialAttention(self.dim)
        self.CRU = ChannelAttention(self.in_dim)
        self.shuffle = ChannelShuffle(self.in_dim)

    def forward(self, x):
        x0 = x
        x_left, x_right = torch.split(x, [self.dim, self.dim], dim=1)

        x_left, x_right = self.x_left(x_left), self.x_right(x_right)

        x_left, x_right = self.SRU(x_left), self.SRU(x_right)

        x = torch.cat((x_left, x_right), dim=1)
        # print(x.shape, x0.shape)

        x = self.CRU(x) + x0
        x = self.shuffle(x)

        return x


if __name__ == '__main__':

    model = DCEU(256,2)
    model.eval()
    images = torch.randn(1, 256, 32, 32)
    with torch.no_grad():
        x = model.forward(images)
    print(x.shape)
    flops, params = profile(model, (images,))
    print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))