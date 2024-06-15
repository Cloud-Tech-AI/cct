from typing import Optional
import torch
import torch.nn as nn

from .cbam import CBAM


class ConvTokenizer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_conv_layers: int,
        intermediate_conv_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pooling_kernel_size: int,
        pooling_stride: int,
        pooling_padding: int,
        cbam: bool,
        activation: Optional[nn.Module],
        max_pool: bool,
        conv_bias: bool,
    ):
        super().__init__()

        conv_channels = (
            [input_channels]
            + [intermediate_conv_dim for _ in range(num_conv_layers - 1)]
            + [output_channels]
        )

        conv_channel_pairs = zip(conv_channels[:-1], conv_channels[1:])

        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=(kernel_size, kernel_size),
                        stride=(stride, stride),
                        padding=(padding, padding),
                        bias=conv_bias,
                    ),
                    nn.Identity() if not exists(activation) else activation(),
                    (
                        nn.MaxPool2d(
                            kernel_size=pooling_kernel_size,
                            stride=pooling_stride,
                            padding=pooling_padding,
                        )
                        if max_pool
                        else nn.Identity()
                    ),
                    CBAM(channels_out) if cbam else nn.Identity(),
                )
                for channels_in, channels_out in conv_channel_pairs
            ]
        )
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
