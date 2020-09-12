import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        res_scale,
        padding_mode,
        n_blocks,
        block_idx=None,
    ):
        super(ResBlock, self).__init__()

        padding = kernel_size // 2

        self.block_idx = block_idx
        self.res_scale = res_scale
        
        self.fst_quar = n_blocks // 4
        self.snd_quar = self.fst_quar * 2
        self.thd_quar = self.fst_quar * 3
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
                bias=True,
            ),
        )

    def forward(self, x):
        out = self.body(x).mul(self.res_scale)
        return out + x
        
class ResNet(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size=3,
        n_layers=67,
        filters=256,
        padding_mode='zeros',
    ):
        super(ResNet, self).__init__()
        padding = kernel_size // 2

        # initial convolution
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        # body loops for `n_layers-3` to account for initial + final convs
        # and divide by 2 bc each resblock has 2 convolution layers
        n_blocks = (n_layers - 3) // 2
        self.body = nn.Sequential(
            *[
                ResBlock(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    res_scale=0.1,
                    n_blocks=n_blocks,
                    block_idx=block_idx,
                    padding_mode=padding_mode,
                ) for block_idx in range(n_blocks)
            ],
            nn.Conv2d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        # tail convolution
        self.tail = nn.Sequential(
            nn.Linear(
                in_features=filters,
                out_features=2,
            ),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        pool = nn.AvgPool2d(
            kernel_size=(512, 512),
            stride=None,
            padding=0,
        )
        p = pool(res)[:, :, 0, 0]
        return self.tail(p)
