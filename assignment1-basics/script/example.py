import math

import torch
from torch import nn


# nn.Module是PyTorch 所有“神经网络层”的基础类，继承它之后，PyTorch 才能自动管理参数
class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()   # 调用父类

        self.in_features = in_features
        self.out_features = out_features
        # 告诉Pytorch，这是要训练的参数，自动出现在model.parameters(),model.state_dict()里面（nn.Parameter）
        self.weight = nn.Parameter(
            # 创建一个Tensor
            torch.empty(
                out_features,
                in_features,
                device=device,  # 权重存在哪里
                dtype=dtype,  # 数据类型
            )
        )
        # 按照正态分布给它填上一些较小的随机数，self.weight
        sigma = math.sqrt(
            2 / (in_features + out_features)
        )


        #  截断正态分布的初始化
        # trunc_normal_表示原地修改Tensor，这里表示直接替换self.weight里的值
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,  #均值
            std=sigma,   # 标准差
            a=-3 * sigma,
            b=3 * sigma,
        )
        #  直接写linear=Linear(4,3)初始化类，再y=linear(x)直接调用forward，通过nn.Module
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T