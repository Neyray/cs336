import torch
from torch import Tensor

def cross_entropy(
    input:torch.Tensor,
    target:torch.Tensor,
):
    '''
    input.shape(...,vocab_size)
    target.shape(...)
    '''
    shifted_input=input-torch.max(input,dim=-1,keepdim=True).values

    log_sum_exp=torch.log(
        torch.exp(shifted_input.sum(dim=-1))
    )

    target_logits=shifted_input.gather(
        dim=-1,
        index=target.unsqueeze(-1)
    ).squeeze(-1)


    loss=target_logits-log_sum_exp

    return loss.mean()