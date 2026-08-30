import math

from collections.abc import Iterable
import torch

def gradient_clipping(
    parameters:Iterable[torch.nn.Parameter],
    max_l2_norm:float,
)->None:
    parameters=[
        p
        for p in parameters
        if p.grad is not None
    ]

    if len(parameters)==0:
        return
    
    total_norm_squared=0.0
    for p in parameters:
        grad_norm=p.grad.norm(2).items
        total_norm_squared+=grad_norm**2

    total_norm=math.sqrt(total_norm_squared)

    eps=1e-6
    clip_coef=max_l2_norm/(total_norm+eps)

    if clip_coef<1.0:
        for p in parameters:
            p.grad.mul_(clip_coef)