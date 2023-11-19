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
from torch import nn, Tensor

from module import ResidualBlock

__all__ = [
    "Discriminator", "discriminator",
]


class Discriminator(nn.Module):
    def __init__(
            self,
            img_size: int = 256,
            num_domains: int = 5,
            channels: int = 64,
            num_blocks: int = 5,
    ) -> None:
        """Discriminator of the StarGAN v2.

        Args:
            img_size (int, optional): The size of the input image. Defaults: 256.
            num_domains (int, optional): The number of domains. Defaults: 5.
            channels (int, optional): The number of channels in all conv blocks. Defaults: 64.
            num_blocks (int, optional): The number of conv blocks in the discriminator. Defaults: 5.

        """
        super(Discriminator, self).__init__()
        self.img_size = img_size
        if img_size != 256:
            raise NotImplementedError("Discriminator currently only supports img_size=256")

        main = [nn.Conv2d(3, channels, 3, 1, 1)]

        curr_channels = channels
        for _ in range(0, num_blocks):
            main.append(ResidualBlock(curr_channels, int(curr_channels * 2), downsample=True))
            curr_channels = int(curr_channels * 2)

        main.append(nn.LeakyReLU(0.2))
        main.append(nn.Conv2d(curr_channels, curr_channels, 4, 1, 0))
        main.append(nn.LeakyReLU(0.2))
        main.append(nn.Conv2d(curr_channels, num_domains, 1, 1, 0))

        self.main = nn.Sequential(*main)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.main(x)
        x = x.view(x.size(0), -1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        x = x[idx, y]

        return x


def discriminator(img_size: int = 256, num_domains: int = 2, **kwargs) -> Discriminator:
    return Discriminator(img_size=img_size, num_domains=num_domains, **kwargs)
