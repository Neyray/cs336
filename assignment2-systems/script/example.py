import argparse
import timeit
from contextlib import nullcontext

import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM


# ============================================================
# Model configurations
# CS336 Assignment 2 - 2026 Table 1
# ============================================================

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
    "xl": {
        "d_model": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "d_ff": 10240,
    },
    "10B": {
        "d_model": 4608,
        "num_layers": 50,
        "num_heads": 36,
        "d_ff": 12288,
    },
}


# ============================================================
# CUDA synchronization
# ============================================================

def synchronize(device: torch.device) -> None:
    """
    CUDA operations are asynchronous.

    Before measuring time, synchronize CPU and GPU so that the
    measured time actually contains the GPU computation.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================
# Benchmark
# ============================================================

def benchmark(
    model_size: str,
    batch_size: int = 4,
    context_length: int = 512,
    vocab_size: int = 10000,
    warmup_steps: int = 5,
    measurement_steps: int = 10,
    mode: str = "forward",
    precision: str = "fp32",
    lr: float = 1e-3,
):
    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = MODEL_CONFIGS[model_size]

    print("================================")
    print(f"device            = {device}")
    print(f"model             = {model_size}")
    print(f"precision         = {precision}")
    print(f"batch_size        = {batch_size}")
    print(f"context_length    = {context_length}")
    print(f"vocab_size        = {vocab_size}")
    print(f"mode              = {mode}")
    print(f"warmup_steps      = {warmup_steps}")
    print(f"measurement_steps = {measurement_steps}")
    print(f"config            = {config}")
    print("================================")

    # --------------------------------------------------------
    # Mixed precision context
    # --------------------------------------------------------

    def autocast_context():
        """
        FP32:
            do nothing

        BF16:
            automatically use BF16 for operations such as
            Linear / matrix multiplication when appropriate.
        """

        if precision == "bf16":
            if device.type != "cuda":
                raise RuntimeError(
                    "BF16 benchmark should be run on a CUDA GPU."
                )

            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "The current GPU does not support BF16."
                )
            # 进入这个代码区域以后，PyTorch 自动判断每个运算到底应该使用 BF16 还是 FP32
            return torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )

        # FP32: no autocast
        return nullcontext()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    # --------------------------------------------------------
    # Random input tokens
    #
    # x.shape = (batch_size, context_length)
    # --------------------------------------------------------

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
    )

    # --------------------------------------------------------
    # Random target tokens
    #
    # y.shape = (batch_size, context_length)
    # --------------------------------------------------------

    y = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    # ========================================================
    # One benchmark step
    # ========================================================

    def run_step():
        # 清空上一轮梯度
        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------
        # Mixed precision only wraps:
        #
        # forward
        # +
        # loss
        #
        # backward 放在 autocast 外
        # ----------------------------------------------------

        with autocast_context():
            # logits.shape:
            #
            # (B, T, vocab_size)
            logits = model(x)

            # ------------------------------
            # Forward only
            # ------------------------------
            if mode == "forward":
                return

            # ------------------------------
            # Compute loss
            #
            # logits:
            # (B, T, V)
            #
            # ->
            #
            # (B*T, V)
            #
            # y:
            # (B, T)
            #
            # ->
            #
            # (B*T)
            # ------------------------------

            loss = loss_fn(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )

        # ----------------------------------------------------
        # backward 不放在 autocast context 中
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Full training step
        # ----------------------------------------------------

        if mode == "train":
            optimizer.step()

    # ========================================================
    # Warm-up
    # ========================================================

    print(f"\nWarm-up: {warmup_steps} steps")

    for _ in range(warmup_steps):
        run_step()

        # 等 GPU 真正执行完
        synchronize(device)

    # ========================================================
    # Measurement
    # ========================================================

    times = []

    print(f"Benchmark: {measurement_steps} steps")

    for _ in range(measurement_steps):

        # ------------------------------
        # 等待之前 GPU 工作完成
        # ------------------------------

        synchronize(device)

        start = timeit.default_timer()

        run_step()

        # ------------------------------
        # 等待当前 GPU 工作完成
        # ------------------------------

        synchronize(device)

        end = timeit.default_timer()

        times.append(end - start)

    # ========================================================
    # Statistics
    # ========================================================

    times = np.array(times)

    mean = times.mean()
    std = times.std()

    print("\n==============================")
    print(f"model     : {model_size}")
    print(f"precision : {precision}")
    print(f"mode      : {mode}")
    print(f"mean      : {mean * 1000:.3f} ms")
    print(f"std       : {std * 1000:.3f} ms")
    print("==============================")

    print("\nraw timings (ms):")

    for i, t in enumerate(times):
        print(f"{i:02d}: {t * 1000:.3f}")

    return times


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Model size
    # --------------------------------------------------------

    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS.keys(),
        default="small",
    )

    # --------------------------------------------------------
    # Benchmark mode
    #
    # forward:
    #     only model(x)
    #
    # forward_backward:
    #     model(x) + loss + backward
    #
    # train:
    #     model(x) + loss + backward + optimizer.step()
    # --------------------------------------------------------

    parser.add_argument(
        "--mode",
        choices=[
            "forward",
            "forward_backward",
            "train",
        ],
        default="forward",
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    parser.add_argument(
        "--precision",
        choices=[
            "fp32",
            "bf16",
        ],
        default="fp32",
    )

    # --------------------------------------------------------
    # Input configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
    )

    # --------------------------------------------------------
    # Benchmark configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--measurement-steps",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    benchmark(
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        mode=args.mode,
        precision=args.precision,
    )