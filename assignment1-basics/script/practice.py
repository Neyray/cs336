import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(
        self,
        theta:int,
        d_k:int,
        max_seq_len:int,
        device:torch.device | None=None,
    ):
        positions=torch.arrange(max_seq_len,device=device)

        freqs=1.0/(
            theta**(torch.arrange(0,d_k,2,device=device)/d_k).float()
        )

        angles=torch.outer(positions,freqs)

        self.register_buffer(
            "cos",
            torch.cos(angles)
        )

        self.register_buffer(
            "sin",
            torch.sin(angles)
        )

    def forward(
        self,
        token_position:torch.Tensor,
        x:torch.Tensor
    )->torch.Tensor:
        
        cos=self.cos[token_position]
        sin=self.sin[token_position]

        x_even=x[...,::2]
        x_odd=x[...,1::2]

        rotated_even=(x_even*cos-x_odd*sin)
        rotated_odd=(x_even*sin+x_odd*cos)

        rotated=torch.stack(
            [x_even,x_odd],
            dim=-1
        )
        return rotated.flatten(-2)