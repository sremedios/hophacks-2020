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
    
class UpConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        padding,
        kernel_size=3,
    ):
        super(UpConv, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )
        
        self.unpool = nn.MaxUnpool2d(kernel_size=2)
        
        self.body = nn.Sequential(
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
        
    def forward(self, x, indices):
        x_up = self.unpool(self.head(x), indices)
        return self.body(x_up)
        

class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size=3,
        ds=1,
    ):
        super(UNet, self).__init__()

        padding = kernel_size // 2


        self.downsample = nn.MaxPool2d(kernel_size=2, return_indices=True)
        
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

        self.upsample_4 = UpConv(
            in_channels=1024//ds,
            out_channels=512//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        self.up_4 = ConvBlock(
            in_channels=1024//ds,
            out_channels=512//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        self.upsample_3 = UpConv(
            in_channels=512//ds,
            out_channels=256//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        self.up_3 = ConvBlock(
            in_channels=512//ds,
            out_channels=256//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        
        self.upsample_2 = UpConv(
            in_channels=256//ds,
            out_channels=128//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        self.up_2 = ConvBlock(
            in_channels=256//ds,
            out_channels=128//ds,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        self.upsample_1 = UpConv(
            in_channels=128//ds,
            out_channels=64//ds,
            kernel_size=kernel_size,
            padding=padding,
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
        
    def extract_features(self, x):
        stage_1 = self.down_1(x)
        stage_1_pool, stage_1_pool_idx = self.downsample(stage_1)
        
        stage_2 = self.down_2(stage_1_pool)
        stage_2_pool, stage_2_pool_idx = self.downsample(stage_2)
        
        stage_3 = self.down_3(stage_2_pool)
        stage_3_pool, stage_3_pool_idx = self.downsample(stage_3)
        
        stage_4 = self.down_4(stage_3_pool)
        stage_4_pool, stage_4_pool_idx = self.downsample(stage_4)

        bn = self.bn(stage_4_pool)
        
        stage_4_up = self.upsample_4(bn, stage_4_pool_idx)
        stage_4_cat = torch.cat([stage_4, stage_4_up], dim=1)
        stage_4 = self.up_4(stage_4_cat)
        
        stage_3_up = self.upsample_3(stage_4, stage_3_pool_idx)
        stage_3_cat = torch.cat([stage_3, stage_3_up], dim=1)
        stage_3 = self.up_3(stage_3_cat)
        
        stage_2_up = self.upsample_2(stage_3, stage_2_pool_idx)
        stage_2_cat = torch.cat([stage_2, stage_2_up], dim=1)
        stage_2 = self.up_2(stage_2_cat)
        
        stage_1_up = self.upsample_1(stage_2, stage_1_pool_idx)
        stage_1_cat = torch.cat([stage_1, stage_1_up], dim=1)
        stage_1 = self.up_1(stage_1_cat)
        
        return stage_1
    
    def encode(self, x):
        y = self.extract_features(x)
        
        pool = nn.AvgPool2d(
            kernel_size=y.shape[-2:],
            stride=None,
            padding=0,
        )
        return pool(y)
    
    def forward(self, x):
        stage_1 = self.extract_features(x)

        return self.seg(stage_1)
