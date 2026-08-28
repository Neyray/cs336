import torch
import math

def softmax(
    x:torch.Tensor,
    dim:int
)->torch.Tensor:
    x_max=torch.max(x,dim=dim,keepdim=True).values
    x_shift=x-x_max

    return torch.exp(x_shift)/torch.sum(torch.exp(x_shift),dim=dim,keepdim=True)

def scaled_dot_attention_product(
    query:torch.Tensor,
    key:torch.Tensor,
    value:torch.Tensor,
    mask:torch.Tensor | None = None
)-> torch.Tensor:
    # 取 query 最后一维的大小
    d_k=query.shape[-1]

    scores=torch.einsum(
        "...q d,...k d -> ...q k",
        query,
        key,
    )

    scores=scores/torch.sqrt(d_k)

    if mask is not None:
        scores=torch.masked_fill(scores,~mask,float("-inf"))

    attn=softmax(scores,dim=-1)# 对最后一维计算出来的分数进行归一化处理

    output=torch.einsum(
        "...q k,...k d -> ...q d",
        attn,
        value,
    )
    return output