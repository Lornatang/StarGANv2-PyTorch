# Copyright 2023 Lornatang Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F_torch

__al__ = [
    "AddCoordsTh", "ConvBlock", "CoordConvTh", "HourGlass", "ResidualBlock",
]

class AddCoordsTh(nn.Module):
    def __init__(self, height=64, width=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            x_coords = torch.arange(height).unsqueeze(1).expand(height, width).float()
            y_coords = torch.arange(width).unsqueeze(0).expand(height, width).float()
            x_coords = (x_coords / (height - 1)) * 2 - 1
            y_coords = (y_coords / (width - 1)) * 2 - 1
            coords = torch.stack([x_coords, y_coords], dim=0)  # (2, height, width)

            if self.with_r:
                rr = torch.sqrt(torch.pow(x_coords, 2) + torch.pow(y_coords, 2))  # (height, width)
                rr = (rr / torch.max(rr)).unsqueeze(0)
                coords = torch.cat([coords, rr], dim=0)

            self.coords = coords.unsqueeze(0).to(device)  # (1, 2 or 3, height, width)
            self.x_coords = x_coords.to(device)
            self.y_coords = y_coords.to(device)

    def forward(self, x, heatmap=None):
        """
        x: (batch, c, x_dim, y_dim)
        """
        coords = self.coords.repeat(x.size(0), 1, 1, 1)

        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(self.x_coords)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, self.x_coords, zero_tensor).to(zero_tensor.device)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, self.y_coords, zero_tensor).to(zero_tensor.device)
            coords = torch.cat([coords, xx_boundary_channel, yy_boundary_channel], dim=1)

        x_and_coords = torch.cat([x, coords], dim=1)
        return x_and_coords


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        self.downsample = None
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.BatchNorm2d(in_planes),
                                            nn.ReLU(True),
                                            nn.Conv2d(in_planes, out_planes, 1, 1, bias=False))


class CoordConvTh(nn.Module):
    def __init__(
            self,
            height,
            width,
            with_r,
            with_boundary,
            in_channels,
            first_one=False,
            *args,
            **kwargs,
    ) -> None:
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(height, width, with_r, with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(64,
                                     64,
                                     True,
                                     True,
                                     256,
                                     first_one,
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module("b1_" + str(level), ConvBlock(256, 256))
        self.add_module("b2_" + str(level), ConvBlock(256, 256))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module("b2_plus_" + str(level), ConvBlock(256, 256))
        self.add_module("b3_" + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules["b1_" + str(level)](up1)
        low1 = F_torch.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules["b2_" + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules["b2_plus_" + str(level)](low2)
        low3 = low2
        low3 = self._modules["b3_" + str(level)](low3)
        up2 = F_torch.interpolate(low3, scale_factor=2, mode="nearest")

        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


class ResidualBlock(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            actv=nn.LeakyReLU(0.2),
            normalize=False,
            downsample=False,
    ) -> None:
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F_torch.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F_torch.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance
