from model.blocks.DCEU import *


class Detail_branch(nn.Module):
    def __init__(self, dim=[32, 64, 128, 256, 512], dilatiaon = [2, 2, 2, 2]):
        super(Detail_branch, self).__init__()

        self.in_dims = dim
        self.dilation = dilatiaon

        self.detail_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels=self.in_dims[0], out_channels=self.in_dims[1], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.in_dims[1]),
            nn.ReLU(inplace=True),

            DCEU(self.in_dims[1], self.dilation[0])
        )

        self.detail_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dims[1], out_channels=self.in_dims[2], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.in_dims[2]),
            nn.ReLU(inplace=True),

            DCEU(self.in_dims[2], self.dilation[1]),
        )

        self.detail_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dims[2], out_channels=self.in_dims[3], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.in_dims[3]),
            nn.ReLU(inplace=True),

            DCEU(self.in_dims[3], self.dilation[2]),
        )

        self.detail_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dims[3], out_channels=self.in_dims[4], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.in_dims[4]),
            nn.ReLU(inplace=True),

            DCEU(self.in_dims[4], self.dilation[3]),
        )

    def forward(self, x):

        detail_1 = self.detail_1(x)
        detail_2 = self.detail_2(detail_1)
        detail_3 = self.detail_3(detail_2)
        detail_4 = self.detail_4(detail_3)

        outputs = []
        outputs.append(detail_1)
        outputs.append(detail_2)
        outputs.append(detail_3)
        outputs.append(detail_4)

        return outputs


if __name__ == '__main__':
    model = Detail_branch()
    images = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        outputs = model.forward(images)
    print(outputs[0].shape, outputs[1].shape, outputs[2].shape, outputs[3].shape)
    flops, params = profile(model, (images,))
    print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))




