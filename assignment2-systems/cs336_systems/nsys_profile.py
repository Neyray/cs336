import argparse
import math
import timeit

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from einops import einsum

import cs336_basics.model as model_module
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.optimizer import AdamW


MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
    },
    "medium": {
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
    },
    "large": {
        "d_model": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "d_ff": 5120,
    },
}


# ---------------------------------------------------------
# 给 Attention 内部操作加 NVTX 标签
# ---------------------------------------------------------

def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:

    d_k = K.shape[-1]

    with nvtx.range("attention/scores"):
        attention_scores = (
            einsum(
                Q,
                K,
                "... query d_k, ... key d_k -> ... query key",
            )
            / math.sqrt(d_k)
        )

    if mask is not None:
        with nvtx.range("attention/mask"):
            attention_scores = torch.where(
                mask,
                attention_scores,
                float("-inf"),
            )

    with nvtx.range("attention/softmax"):
        attention_weights = softmax(
            attention_scores,
            dim=-1,
        )

    with nvtx.range("attention/value"):
        output = einsum(
            attention_weights,
            V,
            "... query key, ... key d_v -> ... query d_v",
        )

    return output


# ---------------------------------------------------------
# 普通 step：专门用于 warm-up
# ---------------------------------------------------------

def warmup_step(
    model,
    optimizer,
    x,
    y,
    vocab_size,
    mode,
):

    optimizer.zero_grad(set_to_none=True)

    logits = model(x)

    if mode == "forward":
        return

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
    )

    loss.backward()

    if mode == "train":
        optimizer.step()


# ---------------------------------------------------------
# 真正要被 Nsight 捕获的一个 step
# ---------------------------------------------------------

def profile_step(
    model,
    optimizer,
    x,
    y,
    vocab_size,
    mode,
):
    optimizer.zero_grad(set_to_none=True)

    timings = {}

    # nsys 只捕获这个 range
    with nvtx.range("profile/measure"):

        # ------------------
        # Forward
        # ------------------
        with nvtx.range("forward"):
            start = timeit.default_timer()

            logits = model(x)

            torch.cuda.synchronize()
            timings["forward"] = timeit.default_timer() - start

        if mode == "forward":
            return timings

        # ------------------
        # Loss
        # ------------------
        with nvtx.range("loss"):
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )
            torch.cuda.synchronize()

        # ------------------
        # Backward
        # ------------------
        with nvtx.range("backward"):
            start = timeit.default_timer()

            loss.backward()

            torch.cuda.synchronize()
            timings["backward"] = timeit.default_timer() - start

        # ------------------
        # AdamW
        # ------------------
        if mode == "train":
            with nvtx.range("optimizer"):
                start = timeit.default_timer()

                optimizer.step()

                torch.cuda.synchronize()
                timings["optimizer"] = timeit.default_timer() - start

    return timings


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS.keys(),
        default="small",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "forward",
            "forward_backward",
            "train",
        ],
        default="train",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Nsight profiling requires CUDA.")

    device = torch.device("cuda")

    torch.manual_seed(0)

    config = MODEL_CONFIGS[args.model_size]

    print(f"model          = {args.model_size}")
    print(f"context length = {args.context_length}")
    print(f"batch size     = {args.batch_size}")
    print(f"mode           = {args.mode}")
    print(f"config         = {config}")

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    # 使用你自己的 AdamW
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    x = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )

    y = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    print(f"\nWarm-up: {args.warmup_steps} steps")

    for _ in range(args.warmup_steps):
        warmup_step(
            model,
            optimizer,
            x,
            y,
            args.vocab_size,
            args.mode,
        )

    torch.cuda.synchronize()

    # warm-up 完成以后再替换 Attention
    # 避免 warm-up 的 attention NVTX 混进正式统计
    model_module.scaled_dot_product_attention = (
        annotated_scaled_dot_product_attention
    )

    # -----------------------------------------------------
    # 只 profile 一个稳定 step
    # -----------------------------------------------------

    timings = profile_step(
        model,
        optimizer,
        x,
        y,
        args.vocab_size,
        args.mode,
    )

    print("\n==========================")

    for name, value in timings.items():
        print(f"{name:10s}: {value * 1000:.3f} ms")

    print("==========================")


if __name__ == "__main__":
    main()