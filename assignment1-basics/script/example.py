import math
from collections.abc import Iterable

import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """
    Clip the global L2 norm of all parameter gradients.

    The gradients are modified in-place.
    """

    # parameters 可能是 model.parameters() 这样的迭代器，
    # 后面需要遍历多次，所以先转成 list。
    parameters = [
        p for p in parameters
        if p.grad is not None
    ]

    if len(parameters) == 0:
        return

    # 1. 计算所有参数梯度合起来的全局 L2 norm
    total_norm_squared = 0.0

    for p in parameters:
        grad_norm = p.grad.norm(2).item()
        total_norm_squared += grad_norm ** 2

    total_norm = math.sqrt(total_norm_squared)

    # 2. 计算裁剪比例
    eps = 1e-6
    clip_coef = max_l2_norm / (total_norm + eps)

    # 3. norm 太大时，所有梯度乘同一个系数
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.mul_(clip_coef)