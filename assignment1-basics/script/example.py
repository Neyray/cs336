import math
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # 取 query 最后一维的大小
    d_k = query.shape[-1]

    scores = torch.einsum(
        "... q d, ... k d -> ... q k",
        query,
        key,
    )

    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)

    output = torch.einsum(
        "... q k, ... k d -> ... q d",
        attn,
        value,
    )

    return output