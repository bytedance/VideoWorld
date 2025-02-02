# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch import nn
import torch

class BaseConnector(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self._in_channels == self._out_channels
        return x


class ConvConnector(BaseConnector):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conv = nn.Conv2d(
            self._in_channels,
            self._out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(x)