import torch
from torch import nn
from jaxtyping import Float, Int
from torch import Tensor


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # 1. token ID -> d_model 向量
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # 2. 堆叠 num_layers 个 Transformer Block
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. 所有 Transformer Block 之后再做一次 RMSNorm
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        # 4. d_model -> vocab_size
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_indices: Int[Tensor, "batch_size sequence_length"],
    ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:

        # (B,T)
        #     ↓ Embedding
        # (B,T,d_model)
        x = self.token_embeddings(in_indices)

        # 每一层都保持：
        # (B,T,d_model) -> (B,T,d_model)
        for layer in self.layers:
            x = layer(x)

        # (B,T,d_model)
        x = self.ln_final(x)

        # (B,T,d_model)
        #       ↓
        # (B,T,vocab_size)
        logits = self.lm_head(x)

        return logits