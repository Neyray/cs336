import math

import torch
from torch import nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features:int,
        out_features:int,
        device:torch.device | None=None,
        dtype:torch.dtype | None=None,
    ):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features

        # 创建一个系统权重
        self.weight=nn.parameter(
            torch.empty(
                out_features,
                in_features,
                device,
                dtype,
            )
        )

        sigma=math.sqrt(
            2/(in_features+out_features)
        )

        # 对权重进行分配数字，正态分布截断
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma,
        )


    def forward(self,x:torch.Tensor)->torch.Tensor:
        return x @ (self.weight.T)