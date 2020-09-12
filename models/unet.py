import torch.nn as nn

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
        n_layers=67,
        ds=1,
    ):
        padding = kernel_size // 2
        super(UNet, self).__init__()
        
        self.downsample = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.MaxUnpool2d(kernel_size=2)


        self.down_1 = ConvBlock(in_channels=n_channels,out_channels=64//ds,padding=padding)
        self.down_2 = ConvBlock(in_channels=64//ds,out_channels=128//ds,padding=padding)
        self.down_3 = ConvBlock(in_channels=128//ds,out_channels=256//ds,padding=padding)
        self.down_4 = ConvBlock(in_channels=256//ds,out_channels=512//ds,padding=padding)
        
        self.bn = ConvBlock(in_channels=512//ds,out_channels=1024//ds,padding=padding)
        
        self.up_4 = ConvBlock(in_channels=1024//ds,out_channels=512//ds,padding=padding)
        self.up_3 = ConvBlock(in_channels=512//ds,out_channels=256//ds,padding=padding)
        self.up_2 = ConvBlock(in_channels=256//ds,out_channels=128//ds,padding=padding)
        self.up_1 = ConvBlock(in_channels=128//ds,out_channels=64//ds,padding=padding)
        
        self.seg = nn.Sequential(
            nn.Conv2d(
                in_channels=64//ds,
                out_channels=n_channels,
                kernel_size=1,
                padding=padding,
                stride=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    
    def encode(self, x):
        stage_1 = self.down_1(x)
        stage_2 = self.down_2(self.downsample(stage_1))
        stage_3 = self.down_2(self.downsample(stage_2))
        stage_4 = self.down_2(self.downsample(stage_3))
        
        bn = self.bn(stage_4)
        
        stage_4 = self.up_4(torch.cat([bn, self.upsample(stage_4)], dim=2))
        stage_3 = self.up_3(torch.cat([stage_3, self.upsample(stage_4)], dim=2))
        stage_2 = self.up_2(torch.cat([stage_2, self.upsample(stage_3)], dim=2))
        stage_1 = self.up_1(torch.cat([stage_1, self.upsample(stage_2)], dim=2))
        
        return stage_1
    
    def forward(self, x):
        return self.seg(self.encode(x))
