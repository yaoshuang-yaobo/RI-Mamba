from torch import nn
import torch
from thop import profile
import torch.nn.functional as F

# 自定义 GroupBatchnorm2d 类，实现分组批量归一化


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, group_num:int = 16, eps:float = 1e-10):
        super(GroupBatchnorm2d,self).__init__()  # 调用父类构造函数
        assert c_num >= group_num  # 断言 c_num 大于等于 group_num
        self.group_num  = group_num  # 设置分组数量
        self.gamma      = nn.Parameter(torch.randn(c_num, 1, 1))  # 创建可训练参数 gamma
        self.beta       = nn.Parameter(torch.zeros(c_num, 1, 1))  # 创建可训练参数 beta
        self.eps        = eps  # 设置小的常数 eps 用于稳定计算
    def forward(self, x):
        N, C, H, W  = x.size()  # 获取输入张量的尺寸
        x           = x.view(N, self.group_num, -1)  # 将输入张量重新排列为指定的形状
        mean        = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std         = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x           = (x - mean) / (std + self.eps)  # 应用批量归一化
        x           = x.view(N, C, H, W)  # 恢复原始形状
        return x * self.gamma + self.beta  # 返回归一化后的张量


class GLAM(nn.Module):
    def __init__(self,
                 oup_channels:int,  # 输出通道数
                 group_num:int = 16,  # 分组数，默认为16
                 gate_treshold:float = 0.5,  # 门控阈值，默认为0.5
                 torch_gn:bool = False  # 是否使用PyTorch内置的GroupNorm，默认为False
                 ):
        super().__init__()  # 调用父类构造函数
         # 初始化 GroupNorm 层或自定义 GroupBatchnorm2d 层
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold  = gate_treshold  # 设置门控阈值
        self.sigomid        = nn.Sigmoid()  # 创建 sigmoid 激活函数

    def forward(self, x):
        gn_x        = self.gn(x)  # 应用分组批量归一化
        w_gamma     = self.gn.gamma / sum(self.gn.gamma)  # 计算 gamma 权重
        reweights   = self.sigomid(gn_x * w_gamma)  # 计算重要性权重
        # 门控机制
        info_mask    = reweights >= self.gate_treshold  # 计算信息门控掩码
        noninfo_mask = reweights < self.gate_treshold  # 计算非信息门控掩码

        return info_mask, noninfo_mask


class G2LIM(nn.Module):
    def __init__(self, dim=64):
        super(G2LIM, self).__init__()

        self.dim = dim

        self.up = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.low = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )

        self.GLAM = GLAM(oup_channels=self.dim, group_num=16, gate_treshold=0.5)


    def forward(self, up, low):
        up = self.up(up)
        low = self.low(low)

        up0 = up
        low0 = low

        x = up + low

        info_mask, noninfo_mask = self.GLAM(x)

        up  = up0 * info_mask
        low = low0 * noninfo_mask

        out = self.reconstruct(up, low)

        return out

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 拆分特征为两部分
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 拆分特征为两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 重构特征并连接


if __name__ == '__main__':

    model  = G2LIM(64)
    model.eval()
    up = torch.randn(1, 64, 32, 32)
    low = torch.randn(1, 64, 16, 16)
    with torch.no_grad():
        x = model.forward(up, low)
    print(x.shape)
    flops, params = profile(model, (up, low,))
    print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
