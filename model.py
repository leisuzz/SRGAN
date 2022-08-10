import torch
from torch import nn
import torch.nn as nn
from torchvision.models import vgg19
import config


class ConvBlock(nn.Module):
    def __int__(self,
                in_channels,
                out_channels,
                discriminator=False,
                use_act=True,
                use_bn=True,
                *args,
                **kwargs):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __int__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)  # ( C * r^2, H, W) to (C, H*r, W*r)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv()))


class Residual(nn.Module):
    def __int__(self, in_channels):
        super(Residual, self).__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return x + out  # skip connection


class Generator(nn.Module):
    def __int__(self,
                in_channels=3,
                num_channels=64,
                num_blocks=16):  # k3n64s1
        super(Generator, self).__int__()
        self.initial = ConvBlock(in_channels,
                                 num_channels,
                                 kernel_size=9,
                                 stride=1,
                                 padding=4,
                                 use_bn=False)  # k9n64s1
        self.residuals = nn.Sequential(*[Residual(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels,
                                   num_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   use_act=False)  # k3n64s1
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels * 4, 2),
                                       UpsampleBlock(num_channels * 4, 2))  # k3n256s1
        self.out = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)  # k9n3s1

    def forward(self, x):
        initial = self.initial(x)  # save for skip connection
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.out(x))


class Discriminator(nn.Module):
    def __int__(self,
                in_channels=3,
                features=[64, 64, 128, 128, 256, 256, 512, 512]
                ):
        super(Discriminator, self).__init__()
        blocks = []

        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return nn.Sigmoid(self.classifier(x))

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)



def test():
    low_resolution = 24
    with torch.cuda.amp.autocast():
        x = torch.randn(5, 3, low_resolution, low_resolution)
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
        disc_out = disc(gen_out)

        print(gen.shape)
        print(disc_out.shape)


if __name__ == "__main__":
    test()
