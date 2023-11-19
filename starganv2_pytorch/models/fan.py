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
import torch
from torch import nn
from torch.nn import functional as F_torch

from .module import ConvBlock, CoordConvTh, HourGlass
from starganv2_pytorch.utils.common import get_preds_fromhm, preprocess


class FAN(nn.Module):
    def __init__(
            self,
            num_modules: int = 1,
            end_relu: bool = False,
            num_landmarks: int = 98,
            fname_pretrained: str = None,
    ) -> None:
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu

        # Base part
        self.conv1 = CoordConvTh(256,
                                 256,
                                 True,
                                 False,
                                 in_channels=3,
                                 out_channels=64,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        self.add_module("m0", HourGlass(1, 4, 256, first_one=True))
        self.add_module("top_m_0", ConvBlock(256, 256))
        self.add_module("conv_last0", nn.Conv2d(256, 256, 1, 1, 0))
        self.add_module("bn_end0", nn.BatchNorm2d(256))
        self.add_module("l0", nn.Conv2d(256, num_landmarks + 1, 1, 1, 0))

        if fname_pretrained is not None:
            self.load_pretrained_weights(fname_pretrained)

    def load_pretrained_weights(self, fname):
        if torch.cuda.is_available():
            checkpoint = torch.load(fname)
        else:
            checkpoint = torch.load(fname, map_location=torch.device("cpu"))
        model_weights = self.state_dict()
        model_weights.update({k: v for k, v in checkpoint["state_dict"].items() if k in model_weights})
        self.load_state_dict(model_weights)

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F_torch.relu(self.bn1(x), True)
        x = F_torch.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        boundary_channels = []
        tmp_out = None
        ll, boundary_channel = self._modules["m0"](x, tmp_out)
        ll = self._modules["top_m_0"](ll)
        ll = F_torch.relu(self._modules["bn_end0"](self._modules["conv_last0"](ll)), True)

        # Predict heatmaps
        tmp_out = self._modules["l0"](ll)
        if self.end_relu:
            tmp_out = F_torch.relu(tmp_out)  # HACK: Added relu
        outputs.append(tmp_out)
        boundary_channels.append(boundary_channel)
        return outputs, boundary_channels

    @torch.no_grad()
    def get_heatmap(self, x, b_preprocess=True):
        """ outputs 0-1 normalized heatmap """
        x = F_torch.interpolate(x, size=256, mode="bilinear")
        x_01 = x * 0.5 + 0.5
        outputs, _ = self(x_01)
        heatmaps = outputs[-1][:, :-1, :, :]
        scale_factor = x.size(2) // heatmaps.size(2)
        if b_preprocess:
            heatmaps = F_torch.interpolate(heatmaps, scale_factor=scale_factor,
                                           mode="bilinear", align_corners=True)
            heatmaps = preprocess(heatmaps)
        return heatmaps

    @torch.no_grad()
    def get_landmark(self, x):
        """ outputs landmarks of x.shape """
        heatmaps = self.get_heatmap(x, b_preprocess=False)
        landmarks = []
        for i in range(x.size(0)):
            pred_landmarks = get_preds_fromhm(heatmaps[i].cpu().unsqueeze(0))
            landmarks.append(pred_landmarks)
        scale_factor = x.size(2) // heatmaps.size(2)
        landmarks = torch.cat(landmarks) * scale_factor
        return landmarks
