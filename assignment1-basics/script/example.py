import torch
from torch import Tensor


def cross_entropy(
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    inputs:  (..., vocab_size)
    targets: (...)

    return:
        scalar loss
    """

    # 1. 数值稳定性：每组 logits 减去自己的最大值
    shifted_inputs = inputs - inputs.max(dim=-1, keepdim=True).values

    # 2. log(sum(exp(logits)))
    log_sum_exp = torch.log(
        torch.exp(shifted_inputs).sum(dim=-1)
    )

    # 3. 取出正确 token 对应的 logit
    target_logits = shifted_inputs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # 4. CE = log(sum(exp(logits))) - 正确 token 的 logit
    loss = log_sum_exp - target_logits

    # 对所有 batch-like 维度平均
    return loss.mean()