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
from collections import namedtuple
from copy import deepcopy

import numpy as np
import torch
from munch import Munch
from torch import Tensor

__all__ = [
    "denorm", "get_preds_fromhm", "norm", "preprocess", "resize", "shift", "truncate"
]

IDXPAIR = namedtuple("IDXPAIR", "start end")
index_map = Munch(chin=IDXPAIR(0 + 8, 33 - 8),
                  eyebrows=IDXPAIR(33, 51),
                  eyebrowsedges=IDXPAIR(33, 46),
                  nose=IDXPAIR(51, 55),
                  nostrils=IDXPAIR(55, 60),
                  eyes=IDXPAIR(60, 76),
                  lipedges=IDXPAIR(76, 82),
                  lipupper=IDXPAIR(77, 82),
                  liplower=IDXPAIR(83, 88),
                  lipinner=IDXPAIR(88, 96))
OPPAIR = namedtuple("OPPAIR", "shift resize")


def denorm(x: Tensor or np.ndarray) -> Tensor:
    """Convert the range from [-1, 1] to [0, 1].

    Args:
        x (torch.Tensor or np.ndarray): Input tensor or array.

    Returns:
        torch.Tensor: Output tensor.
    """

    out = (x + 1) / 2
    return out.clamp_(0, 1)


def get_preds_fromhm(hm):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)
    return preds


def norm(x, eps=1e-6):
    """Apply min-max normalization."""
    x = x.contiguous()
    N, C, H, W = x.size()
    x_ = x.view(N * C, -1)
    max_val = torch.max(x_, dim=1, keepdim=True)[0]
    min_val = torch.min(x_, dim=1, keepdim=True)[0]
    x_ = (x_ - min_val) / (max_val - min_val + eps)
    out = x_.view(N, C, H, W)
    return out


def preprocess(x):
    """Preprocess 98-dimensional heatmaps."""
    N, C, H, W = x.size()
    x = truncate(x)
    x = norm(x)

    sw = H // 256
    operations = Munch(chin=OPPAIR(0, 3),
                       eyebrows=OPPAIR(-7 * sw, 2),
                       nostrils=OPPAIR(8 * sw, 4),
                       lipupper=OPPAIR(-8 * sw, 4),
                       liplower=OPPAIR(8 * sw, 4),
                       lipinner=OPPAIR(-2 * sw, 3))

    for part, ops in operations.items():
        start, end = index_map[part]
        x[:, start:end] = resize(shift(x[:, start:end], ops.shift), ops.resize)

    zero_out = torch.cat([torch.arange(0, index_map.chin.start),
                          torch.arange(index_map.chin.end, 33),
                          torch.LongTensor([index_map.eyebrowsedges.start,
                                            index_map.eyebrowsedges.end,
                                            index_map.lipedges.start,
                                            index_map.lipedges.end])])
    x[:, zero_out] = 0

    start, end = index_map.nose
    x[:, start + 1:end] = shift(x[:, start + 1:end], 4 * sw)
    x[:, start:end] = resize(x[:, start:end], 1)

    start, end = index_map.eyes
    x[:, start:end] = resize(x[:, start:end], 1)
    x[:, start:end] = resize(shift(x[:, start:end], -8), 3) + \
                      shift(x[:, start:end], -24)

    # Second-level mask
    x2 = deepcopy(x)
    x2[:, index_map.chin.start:index_map.chin.end] = 0  # start:end was 0:33
    x2[:, index_map.lipedges.start:index_map.lipinner.end] = 0  # start:end was 76:96
    x2[:, index_map.eyebrows.start:index_map.eyebrows.end] = 0  # start:end was 33:51

    x = torch.sum(x, dim=1, keepdim=True)  # (N, 1, H, W)
    x2 = torch.sum(x2, dim=1, keepdim=True)  # mask without faceline and mouth

    x[x != x] = 0  # set nan to zero
    x2[x != x] = 0  # set nan to zero
    return x.clamp_(0, 1), x2.clamp_(0, 1)


def resize(x, p=2):
    """Resize heatmaps."""
    return x ** p


def shift(x, N):
    """Shift N pixels up or down."""
    up = N >= 0
    N = abs(N)
    _, _, H, W = x.size()
    head = torch.arange(N)
    tail = torch.arange(H - N)

    if up:
        head = torch.arange(H - N) + N
        tail = torch.arange(N)
    else:
        head = torch.arange(N) + (H - N)
        tail = torch.arange(H - N)

    # permutation indices
    perm = torch.cat([head, tail]).to(x.device)
    out = x[:, :, perm, :]
    return out


def truncate(x, thres=0.1):
    """Remove small values in heatmaps."""
    return torch.where(x < thres, torch.zeros_like(x), x)
