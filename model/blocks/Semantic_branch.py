from model.blocks.CFM import *


class Semantic_branch(nn.Module):
    def __init__(self, dim=[32, 64, 128, 256, 512]):
        super(Semantic_branch, self).__init__()

        self.in_dims = dim

        self.semantic_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels=self.in_dims[0], out_channels=self.in_dims[1], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.in_dims[1]),
            nn.ReLU(inplace=True),

            CFM(self.in_dims[1])
        )

        self.semantic_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dims[1], out_channels=self.in_dims[2], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.in_dims[2]),
            nn.ReLU(inplace=True),

            CFM(self.in_dims[2])
        )

        self.semantic_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dims[2], out_channels=self.in_dims[3], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.in_dims[3]),
            nn.ReLU(inplace=True),

            CFM(self.in_dims[3])
        )

        self.semantic_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dims[3], out_channels=self.in_dims[4], kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.in_dims[4]),
            nn.ReLU(inplace=True),

            CFM(self.in_dims[4])
        )

    def forward(self, x):

        semantic_1 = self.semantic_1(x)
        semantic_2 = self.semantic_2(semantic_1)
        semantic_3 = self.semantic_3(semantic_2)
        semantic_4 = self.semantic_4(semantic_3)


        outputs = []
        outputs.append(semantic_1)
        outputs.append(semantic_2)
        outputs.append(semantic_3)
        outputs.append(semantic_4)

        return outputs


if __name__ == '__main__':
    model = Semantic_branch()
    images = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        outputs = model.forward(images)
    print(outputs[0].shape, outputs[1].shape, outputs[2].shape, outputs[3].shape)
    flops, params = profile(model, (images,))
    print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))




