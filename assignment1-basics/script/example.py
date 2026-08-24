import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):

    def __init__(
        self,
        theta: float,   # 旋转频率
        d_k: int,   # Q/K/V维度，d_model=num_head*d_head(d_k),d_head,d_k都表示每个头的维度
        max_seq_len: int,  # 模型支持的最大序列长度，也就是token_ids长度
        device=None,
    ):
        super().__init__()

        self.d_k = d_k

        # 频率
        # 给 d_k 维 Q/K 向量中的每一对维度，生成一个不同的旋转速度w_i
        freqs = 1.0 / (
            theta ** (
                torch.arange(0, d_k, 2, device=device).float()
                / d_k
            )
        )
        # 提前计算所有位置的旋转参数
        positions = torch.arange(
            max_seq_len,
            device=device
        ).float()

        # 位置*频率=旋转角度
        # (seq_len, d_k/2)
        # 实际上就是矩阵，(max_seq_len, d_k/2)，输入时直接查self.cos[token_positions]
        # theta=m*w_i，第 m 个 token，在第 i 个二维空间旋转多少角度
        angles = torch.outer(
            positions,
            freqs
        )

        # 保存cache，模型状态，不是训练参数
        # cos:(seq_len,d_k/2),d_k/2个旋转角度，因为每两个参数组成一个组，总共有d_k/2个组
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
        x: torch.Tensor,
        token_positions: torch.Tensor,   # 当前输入 token 的位置编号，0,1,2,3...
    ):

        # x:
        # (..., seq_len, d_k)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]


        # 拆偶数维和奇数维
        x_even = x[..., ::2]   # 负责x坐标
        x_odd = x[..., 1::2]   # 负责y坐标

        rotated_even = (
            x_even * cos
            -
            x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            +
            x_odd * cos
        )

        # 合回来
        out = torch.stack(
            [
                rotated_even,
                rotated_odd
            ],
            dim=-1
        )
        #   展开最后二维
        return out.flatten(-2)