import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
            nn.Unflatten(1, torch.Size([gate_channels, 1, 1])),
        )
        self.adaptive_avgpool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_maxpool_2d = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        # x => B x C x H x W
        channel_avg = self.adaptive_avgpool_2d(x)
        # channel_avg => B x C x 1 x 1
        channel_max = self.adaptive_maxpool_2d(x)
        # channel_max => B x C x 1 x 1
        x_out = self.mlp(channel_avg) + self.mlp(channel_max)
        # x_out => B x C x 1 x 1
        scale = F.sigmoid(x_out)
        # scale => B x C x 1 x 1
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialConv(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int = 7,
    ):
        super(SpatialConv, self).__init__()
        self.conv = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels_out, eps=1e-5, momentum=0.01, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = SpatialConv(2, 1)

    def forward(self, x):
        # x => B x C x H x W
        x_compress = self.compress(x)
        # x_compress => B x 2 x H x W
        x_out = self.spatial(x_compress)
        # x_out => B x 1 x H x W
        scale = F.sigmoid(x_out)
        # scale => B x 1 x H x W
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
    ):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        # x => B x C x H x W
        x = self.channel_gate(x)
        # x => B x C x H x W
        out = self.spatial_gate(x)
        # out => B x C x H x W
        return out
