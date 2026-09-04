import argparse
import timeit

import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from contextlib import nullcontext


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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    memory_profile:bool=False,
    snapshot_path:str="memory_snapshot.pickle",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MODEL_CONFIGS[model_size]

    print(f"device = {device}")
    print(f"model = {model_size}")
    print(config)


    def autocast_context():
        """
        FP32:
            do nothing

        BF16:
            automatically use BF16 for operations such as
            Linear / matrix multiplication when appropriate.
        """

        if precision=="bf16":
            if device.type!="cuda":
                raise RuntimeError("BF16 benchmark should be run on CUDA.")
            
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Current GPU does not support BF16.")
            
            # 进入这个代码区域以后，PyTorch 自动判断每个运算到底应该使用 BF16 还是 FP32
            return torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
        return nullcontext()
    

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    # 随机生成 token IDs
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
    )

    # 随机生成 target token IDs
    y = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    def run_step():
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
            logits=model(x)

            if mode=="forward":return

            loss=loss_fn(
                logits.reshape(-1,vocab_size),
                y.reshape(-1)
            )
        loss.backward()

        if mode=="train":optimizer.step()   



    # -------------------------
    # Warm-up
    # -------------------------

    print(f"\nWarm-up: {warmup_steps} steps")

    for _ in range(warmup_steps):
        run_step()
        synchronize(device)





    # -------------------------
    # Memory Profiling
    # -------------------------
    if memory_profile:
        if device.type!="cuda":
            raise RuntimeError("Memory Profile Needs CUDA")
        
        # 清空之前记录的峰值，避免 warm-up 干扰
        torch.cuda.reset_peak_memory_stats(device)

        # 从这里开始记录 CUDA 显存申请 / 释放历史
        torch.cuda.memory._record_memory_history(
            max_entries=1_000_000
        )


        # 进行一步run_step()
        run_step()
        synchronize(device)

        peak_memory=torch.cuda.max_memory_allocated(device)


        # 保存 snapshot
        torch.cuda.memory._dump_snapshot(snapshot_path)

        # 停止记录
        torch.cuda.memory._record_memory_history(
            enabled=None,
        )

        print("\n==============================")
        print("Memory profiling")
        print(f"model     : {model_size}")
        print(f"context   : {context_length}")
        print(f"mode      : {mode}")
        print(f"precision : {precision}")
        print(f"peak      : {peak_memory / 1024**3:.3f} GiB")
        print(f"snapshot  : {snapshot_path}")
        print("==============================")

        return peak_memory





    # -------------------------
    # Measurement
    # -------------------------

    times = []

    print(f"Benchmark: {measurement_steps} steps")

    for _ in range(measurement_steps):
        # 确保前一个 step 的 GPU 工作已经全部结束
        synchronize(device)

        start = timeit.default_timer()

        run_step()
        # run_step 把 GPU 任务都提交完后，CPU 在这里停下来，直到 GPU 真正完成
        synchronize(device)

        end = timeit.default_timer()

        times.append(end - start)

    times = np.array(times)

    mean = times.mean()
    std = times.std()

    print("\n==============================")
    print(f"mode: {mode}")
    print(f"mean: {mean * 1000:.3f} ms")
    print(f"std : {std * 1000:.3f} ms")
    print("==============================")

    print("\nraw timings (ms):")

    for i, t in enumerate(times):
        print(f"{i:02d}: {t * 1000:.3f}")

    return times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS.keys(),
        default="small",
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32","bf16"],
        default="fp32",
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "train"],
        default="forward",
    )

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10000)

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)

    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Enable CUDA memory profiling.",
    )

    parser.add_argument(
        "--snapshot-path",
        type=str,
        default="memory_snapshot.pickle",
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
        memory_profile=args.memory_profile,
        snapshot_path=args.snapshot_path,
    )