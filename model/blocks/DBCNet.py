import torch
from torch import nn

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SEBlock(nn.Module):
    def __init__(self, mode, channels, ratio):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.global_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v


class crossblock(nn.Module):
    def __init__(self, inc, outc, l=9,ifse=True,norm="gn",group=4):
        super(crossblock, self).__init__()
        self.ifse=ifse
        if norm=="gn":
            self.right1 = nn.Sequential(
                nn.Conv2d(inc, outc // 2, kernel_size=(l, 1), stride=1, padding=(l // 2, 0)),
                nn.GroupNorm(outc // (group*2), outc // 2),
                nn.ReLU(),

            )
            self.right2 = nn.Sequential(
                nn.Conv2d(inc, outc // 2, kernel_size=(1, l), stride=1, padding=(0, l // 2)),
                nn.GroupNorm(outc // (group*2), outc // 2),
                nn.ReLU()
            )
            self.left = nn.Sequential(
                nn.Conv2d(inc, outc // 2, 3, padding=1),
                nn.GroupNorm(outc // (group*2), outc // 2),
                nn.ReLU(inplace=True)
            )
            self.conv = nn.Sequential(
                nn.Conv2d(3 * outc // 2, outc, 3, padding=1),
                nn.GroupNorm(outc // group, outc),
                nn.ReLU(inplace=True)
            )
        elif norm == "bn":
            self.right1 = nn.Sequential(
                nn.Conv2d(inc, outc // 2, kernel_size=(l, 1), stride=1, padding=(l // 2, 0)),
                nn.BatchNorm2d(outc//2),
                nn.ReLU(),

            )
            self.right2 = nn.Sequential(
                nn.Conv2d(inc, outc // 2, kernel_size=(1, l), stride=1, padding=(0, l // 2)),
                nn.BatchNorm2d(outc//2),
                nn.ReLU()
            )
            self.left = nn.Sequential(
                nn.Conv2d(inc, outc // 2, 3, padding=1),
                nn.BatchNorm2d(outc//2),
                nn.ReLU(inplace=True)
            )
            self.conv = nn.Sequential(
                nn.Conv2d(3 * outc // 2, outc, 3, padding=1),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True)
            )

        self.relu = nn.ReLU()
        self.se = SEBlock("avg", 3*outc//2, 2)

    def forward(self, x):
        left = self.left(x)
        right1 = self.right1(x)
        right2 = self.right2(x)
        if self.ifse:
            out = self.se(torch.cat((left,right1, right2), dim=1))
        else:
            out=torch.cat((left,right1, right2), dim=1)
        out = self.conv(out)
        return out


class Sptialbranch(nn.Module):
    def __init__(self,layers,dim,norm='gn',inchannel=3,l=3):
        self.depth=layers
        self.dim=dim
        super().__init__()
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            if i==0:
                self.layers.append(crossblock(inc=inchannel,outc=dim*(2**i),l=l,norm=norm,ifse=False))
            elif i==1:
                self.layers.append(crossblock(inc=dim*(2**(i-1)),outc=dim*(2**i),l=l,norm=norm,ifse=False))
            else:
                self.layers.append(crossblock(inc=dim*(2**(i-1)),outc=dim*(2**i),l=l,norm=norm))
        self.pool=nn.MaxPool2d(2,2)

    def forward(self,x):
        out=[]
        for i,layer in enumerate(self.layers):
            x=layer(x)
            out.append(x)
            if i<self.depth-1:
                x=self.pool(x)
        return out


class Contextbranch(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.embed_dim = 64
        self.inchannel = 3

        self.layer0 = nn.Sequential(
            nn.Conv2d(self.inchannel, self.embed_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim // 4),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 4, self.embed_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim // 4),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.embed_dim // 4, self.embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim * 2 ),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim * 2 , self.embed_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim * 2),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(self.embed_dim * 2, self.embed_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim * 4),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim * 4, self.embed_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim * 4),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(self.embed_dim * 4, self.embed_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim * 8),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim * 8, self.embed_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embed_dim * 8),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(self.pool(x1))
        x3 = self.layer2(self.pool(x2))
        x4 = self.layer3(self.pool(x3))
        x5 = self.layer4(self.pool(x4))
        x6 = self.layer5(self.pool(x5))

        return x1, x2, x3, x4, x5, x6


class FFM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.conv_act=nn.Sequential(
            nn.Conv2d(2*inc,inc,3,1,1),
            nn.GELU()
        )
        self.ca=SEBlock('avg',inc,2)
        self.attforcross=SEBlock('avg',inc,2)
        self.attforvssm=SEBlock('avg',inc,2)
    def forward(self,x_vssm,x_cross):
        x=torch.cat([x_vssm,x_cross],dim=1)
        x=self.conv_act(x)
        x=self.ca(x)
        res_cross=self.attforcross(x_cross)
        res_vssm=self.attforvssm(x_vssm)
        out=x+res_vssm+res_cross
        return out


class Cross_Aware(nn.Module):
    def __init__(self, inc, outc, l=9,ifse=True,norm="gn",group=4):
        super(Cross_Aware, self).__init__()
        self.ifse=ifse
        if norm=="gn":
            self.right1 = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=(l, 1), stride=1, padding=(l // 2, 0)),
                nn.GroupNorm(outc // group, outc),
                nn.ReLU(),

            )
            self.right2 = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=(1, l), stride=1, padding=(0, l // 2)),
                nn.GroupNorm(outc // group, outc),
                nn.ReLU()
            )
            self.left = nn.Sequential(
                nn.Conv2d(inc, outc, 3, padding=1),
                nn.GroupNorm(outc // group, outc),
                nn.ReLU(inplace=True)
            )
            self.conv = nn.Sequential(
                nn.Conv2d(3 * outc, outc, 3, padding=1),
                # nn.GroupNorm(outc // group, outc),
                # nn.ReLU(inplace=True)
            )
        elif norm == "bn":
            self.right1 = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=(l, 1), stride=1, padding=(l // 2, 0)),
                nn.BatchNorm2d(outc),
                nn.ReLU(),

            )
            self.right2 = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=(1, l), stride=1, padding=(0, l // 2)),
                nn.BatchNorm2d(outc),
                nn.ReLU()
            )
            self.left = nn.Sequential(
                nn.Conv2d(inc, outc, 3, padding=1),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True)
            )
            self.conv = nn.Sequential(
                nn.Conv2d(3 * outc, outc, 3, padding=1),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True)
            )

        self.relu = nn.ReLU()
        self.se = SEBlock("avg", 3*outc, 2)

    def forward(self, x):
        left = self.left(x)
        right1 = self.right1(x)
        right2 = self.right2(x)
        if self.ifse:
            out = self.se(torch.cat((left,right1, right2), dim=1))
        else:
            out=torch.cat((left,right1, right2), dim=1)
        out = self.conv(out)
        return out


class Channel_att(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(Channel_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        return x * self.sigmoid(attn)


class CMM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.inc = inc

        self.LN = nn.Sequential(
            nn.Conv2d(self.inc, self.inc, 3, padding=1),
            nn.BatchNorm2d(self.inc),
            nn.SiLU(inplace=True),
        )
        self.Cross = Cross_Aware(self.inc, self.inc, norm='gn')
        self.Channel_att = Channel_att(self.inc)


    def forward(self, x):
        x0 = x
        x = self.LN(x) + x0
        x = self.Channel_att(self.Cross(x)) + x0

        return x


