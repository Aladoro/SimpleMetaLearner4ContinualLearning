import abc

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class OverLayer(abc.ABC):
    """Layer with overridable parameters for inner loop meta-optimization"""
    def __init__(self, over_params):
        self.over_params = over_params
    def forward(self, input: Tensor, params=None) -> Tensor:
        raise NotImplementedError

class OverConv2d(nn.Conv2d, OverLayer):
    def __init__(self, *args, **kwargs):
        nn.Conv2d.__init__(self, *args, **kwargs)
        OverLayer.__init__(self, over_params=2)

    def forward(self, input: Tensor, params=None) -> Tensor:
        if params is not None:
            return self._conv_forward(input, *params)
        else:
            return nn.Conv2d.forward(self, input=input)


class OverLinear(nn.Linear, OverLayer):
    def __init__(self, *args, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        OverLayer.__init__(self, over_params=2)
    def forward(self, input: Tensor, params=None) -> Tensor:
        if params is not None:
            return F.linear(input, *params)
        else:
            return nn.Linear.forward(self, input)

class OverInstanceNorm2d(nn.InstanceNorm2d, OverLayer):
    def __init__(self, *args, **kwargs):
        nn.InstanceNorm2d.__init__(self, *args, **kwargs)
        OverLayer.__init__(self, over_params=2)

    def forward(self, input: Tensor, params=None) -> Tensor:
        if params is not None:
            return self.forward_over(input, *params)
        else:
            return nn.InstanceNorm2d.forward(self, input)

    def forward_over(self, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        self._check_input_dim(input)
        return F.instance_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

class OverBatchNorm2d(nn.BatchNorm2d, OverLayer):
    def __init__(self, *args, **kwargs):
        nn.BatchNorm2d.__init__(self, *args, **kwargs)
        OverLayer.__init__(self, over_params=2)

    def forward(self, input: Tensor, params=None) -> Tensor:
        if params is not None:
            return self.forward_over(input, *params)
        else:
            return nn.BatchNorm2d.forward(self, input)

    def forward_over(self, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        return F.batch_norm(
            input,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )