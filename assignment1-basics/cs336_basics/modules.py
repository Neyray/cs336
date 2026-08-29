import math

import torch
from torch import nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

# nn.Module是PyTorch 所有“神经网络层”的基础类，继承它之后，PyTorch 才能自动管理参数
class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        # : torch.dtype | None   类型标注
        # = None                 默认值
        device:torch.device | None=None,
        dtype:torch.dtype | None=None,
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


    # nn.Module规定，调用对象本身执行forward函数，例，linear=Linear(...),linear.(x)就会调用linear.forward(x)函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings:int, # 词表大小，vocab_size
        embedding_dims:int, # 嵌入向量维度，d_model
        device:torch.device | None =None,
        dtype:torch.dtype | None =None,
    ):
        
        super().__init__()

        #创建一个权重矩阵,vocab_size*d_model，根据ids来选择d_model个数字组成的序列
        self.weight=nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dims,
                # 前面的是size，下面的是关键词参数
                device=device,
                dtype=dtype,
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1,
            a=-3,
            b=3,
        )
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.weight[token_ids]



class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(
            # torch.ones()表示矩阵初始化为1，torch.empty()则为垃圾值
            torch.ones(
                # 这不叫矩阵，torch.Tensor([1,2,3])，刚好和每个token的序列对上，shape(d_model,)
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        x = x.to(torch.float32)

        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True)  # dim=-1表示对最后一个维度进行计算,keepdim=True表示保持结构的不变
            + self.eps
        )

        x = x / rms

        x = x * self.weight

        return x.to(input_dtype)
    


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model:int,
        d_ff:int,   # 进行SwiGLU分析时，需要展开维度，先把shape从(...,d_model)转成(...,d_ff)，再在最后转回来
    ):
        super().__init__()

        self.d_model=d_model
        self.d_ff=d_ff

        self.w1=Linear(d_model,d_ff)
        self.w2=Linear(d_ff,d_model)
        self.w3=Linear(d_model,d_ff)

    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        gate=self.w1(x)
        value=self.w3(x)

        x=nn.functional.silu(gate)# silu部分
        x=x*value# glu部分

        x=self.w2(x)
        
        return x
    

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
    


# softmax：把某一维变成概率分布
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


# 单个attention head的函数
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

    attn = softmax(scores, dim=-1)# 对最后一维计算出来的分数进行归一化处理

    output = torch.einsum(
        "... q k, ... k d -> ... q d",
        attn,
        value,
    )

    return output




# 多头注意力
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta:float | None=None,
        max_seq_len:int | None=None,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 一次性生成 Q, K, V
        # 输入:  (..., d_model)
        # 输出:  (..., 3 * d_model)
        self.qkv_proj = Linear(d_model, 3 * d_model)

        # 多头拼接后再映射回 d_model
        self.output_proj = Linear(d_model, d_model)


        if theta is not None and max_seq_len is not None:
            self.rope=RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
            )
        else:
            self.rope=None

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)

        return:
            (batch_size, seq_len, d_model)
        """

        T=x.shape[-2]

        # qkv: (B, T, 3 * d_model)
        qkv = self.qkv_proj(x)

        # q, k, v: each (B, T, d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # 拆成多头
        # (B, T, d_model)
        # -> (B, num_heads, T, d_head)
        q = rearrange(
            q,
            "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        k = rearrange(
            k,
            "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        v = rearrange(
            v,
            "b t (h d) -> b h t d",
            h=self.num_heads,
        )

        

        if self.rope is not None:
            if token_positions is not None:
                token_positions.unsqueeze(-2)
            else:
                token_positions=torch.arange(
                    T,
                    device=x.device,
                )
            q=self.rope(q,token_positions)
            k=self.rope(k,token_positions)
        
        # causal multi-head attention
        # 输出: (B, num_heads, T, d_head)
        causal_mask=torch.tril(
            torch.ones(T,T,device=x.device,dtype=torch.bool)
        )
        attn_output = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=causal_mask,
        )

        # 多头拼回去
        # (B, num_heads, T, d_head)
        # -> (B, T, d_model)
        attn_output = rearrange(
            attn_output,
            "b h t d -> b t (h d)",
        )

        # 输出投影
        output = self.output_proj(attn_output)

        return output



# Transformer Block
class Transformer_Block(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        theta:int,
        d_ff:int,
        max_seq_len:int,
    ):
        super().__init__()
        self.ln1=RMSNorm(
            d_model=d_model,
        )

        self.attn=MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
        )
        self.ln2=RMSNorm(
            d_model=d_model,
        )

        self.ffn=SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
        )

    
    def forward(
        self,
        x:torch.Tensor,
    ):
        x=x+self.attn(self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x
    



class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        context_length:int,
        d_model:int,
        num_layers:int,
        num_heads:int,
        d_ff:int,
        rope_theta:float,
        device=None,
        dtype=None,
    ):
        
        super().__init__()
        #1.将token_ids转化为向量
        self.token_embeddings=Embedding(
            num_embeddings=vocab_size,
            embedding_dims=d_model,
            device=device,
            dtype=dtype,
        )
        #2.Transformer Block
        self.layers=nn.ModuleList(
            [
                Transformer_Block(
                    d_model=d_model,
                    num_heads=num_heads,
                    theta=rope_theta,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                )
                for _ in range(num_layers)
            ]
        )

        #3.最外层RMSNorm
        self.ln_final=RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        #4.转为vocab_size维
        self.lm_head=Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )



    def forward(
        self,
        in_indices:Int[Tensor,"batch_size sequence_length"],
    )->Float[Tensor,"batch_size sequence_length"]:
        x=self.token_embeddings(in_indices)

        for layer in self.layers:
            x=layer(x)
        x=self.ln_final(x)

        logits=self.lm_head(x)

        return logits