import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        padding,
        kernel_size=3,
    ):
        super(ConvBlock, self).__init__()
        
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.body(x)
        

class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size=3,
        ds=1,
    ):
        super(UNet, self).__init__()

        padding = kernel_size // 2


        self.downsample = nn.MaxPool2d(kernel_size=2)
        
        self.down_1 = ConvBlock(
            in_channels=n_channels,
            out_channels=64//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.down_2 = ConvBlock(
            in_channels=64//ds,
            out_channels=128//ds,
            padding=padding,
        )
        self.down_3 = ConvBlock(
            in_channels=128//ds,
            out_channels=256//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.down_4 = ConvBlock(
            in_channels=256//ds,
            out_channels=512//ds,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.bn = ConvBlock(
            in_channels=512//ds,
            out_channels=1024//ds,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.upsample_4 = nn.ConvTranspose2d(
            in_channels=1024//ds,
            out_channels=512//ds,
            kernel_size=2,
            stride=2,
        )
        self.up_4 = ConvBlock(
            in_channels=1024//ds,
            out_channels=512//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        self.upsample_3 = nn.ConvTranspose2d(
            in_channels=512//ds,
            out_channels=256//ds,
            kernel_size=2,
            stride=2,
        )
        self.up_3 = ConvBlock(
            in_channels=512//ds,
            out_channels=256//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        self.upsample_2 = nn.ConvTranspose2d(
            in_channels=256//ds,
            out_channels=128//ds,
            kernel_size=2,
            stride=2,
        )
        self.up_2 = ConvBlock(
            in_channels=256//ds,
            out_channels=128//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        self.upsample_1 = nn.ConvTranspose2d(
            in_channels=128//ds,
            out_channels=64//ds,
            kernel_size=2,
            stride=2,
        )
        self.up_1 = ConvBlock(
            in_channels=128//ds,
            out_channels=64//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        
        self.seg = nn.Sequential(
            nn.Conv2d(
                in_channels=64//ds,
                out_channels=n_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    
    def _forward(self, x):
        stage_1 = self.down_1(x)
        stage_2 = self.down_2(self.downsample(stage_1))
        stage_3 = self.down_3(self.downsample(stage_2))
        stage_4 = self.down_4(self.downsample(stage_3))

        bn = self.bn(self.downsample(stage_4))
        
        stage_4 = self.up_4(torch.cat([stage_4, self.upsample_4(bn)], dim=1))
        stage_3 = self.up_3(torch.cat([stage_3, self.upsample_3(stage_4)], dim=1))
        stage_2 = self.up_2(torch.cat([stage_2, self.upsample_2(stage_3)], dim=1))
        stage_1 = self.up_1(torch.cat([stage_1, self.upsample_1(stage_2)], dim=1))
        
        return stage_1
    
    def encode(self, x):
        y = self._forward(x)
        pool = nn.AvgPool2d(
            kernel_size=y.shape[-2:],
            stride=None,
            padding=0,
        )
        return pool(y)
    
    def forward(self, x):
        stage_1 = self._forward(x)

        return self.seg(stage_1)