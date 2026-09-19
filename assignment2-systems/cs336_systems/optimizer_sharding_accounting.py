from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM

from cs336_systems.ddp_overlap_individual_parameters import (
    DDPOverlapIndividualParameters,
)

from cs336_systems.sharded_optimizer import ShardedOptimizer


# ============================================================
# 2026 CS336 Assignment 2
# XL configuration
# ============================================================

VOCAB_SIZE = 10_000
BATCH_SIZE = 4
CONTEXT_LENGTH = 512

XL_CONFIG = {
    "d_model": 2560,
    "d_ff": 10240,
    "num_layers": 32,
    "num_heads": 32,
}


# ============================================================
# Memory helpers
# ============================================================

def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def bytes_to_gib(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def get_parameter_memory(model: torch.nn.Module) -> int:
    return sum(
        tensor_bytes(param)
        for param in model.parameters()
    )


def get_gradient_memory(model: torch.nn.Module) -> int:
    return sum(
        tensor_bytes(param.grad)
        for param in model.parameters()
        if param.grad is not None
    )


def get_optimizer_state_memory(
    optimizer: torch.optim.Optimizer,
) -> int:

    # ShardedOptimizer 真正的 optimizer state
    # 在 local_optimizer 里面。
    if isinstance(optimizer, ShardedOptimizer):
        actual_optimizer = optimizer.local_optimizer
    else:
        actual_optimizer = optimizer

    total = 0

    for state in actual_optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += tensor_bytes(value)

    return total


def print_memory_snapshot(
    label: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int,
):
    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)

    parameter_memory = get_parameter_memory(model)
    gradient_memory = get_gradient_memory(model)
    optimizer_memory = get_optimizer_state_memory(optimizer)

    # 防止两个 rank 的输出完全混在一起
    for current_rank in range(dist.get_world_size()):
        dist.barrier()

        if rank == current_rank:
            print()
            print("=" * 70)
            print(f"Rank {rank} | {label}")
            print("=" * 70)

            print(
                f"CUDA allocated:       "
                f"{bytes_to_gib(allocated):.3f} GiB"
            )

            print(
                f"CUDA reserved:        "
                f"{bytes_to_gib(reserved):.3f} GiB"
            )

            print(
                f"CUDA peak allocated:  "
                f"{bytes_to_gib(peak):.3f} GiB"
            )

            print()

            print(
                f"Parameters:           "
                f"{bytes_to_gib(parameter_memory):.3f} GiB"
            )

            print(
                f"Gradients:            "
                f"{bytes_to_gib(gradient_memory):.3f} GiB"
            )

            print(
                f"Optimizer states:     "
                f"{bytes_to_gib(optimizer_memory):.3f} GiB"
            )

            print(flush=True)

    dist.barrier()


# ============================================================
# Training
# ============================================================

def training_step(
    model,
    optimizer,
    x,
    y,
):
    optimizer.zero_grad(set_to_none=True)

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
    )

    loss.backward()

    # Section 5 DDP:
    #
    # 等所有异步 gradient AllReduce 完成
    model.finish_gradient_synchronization()

    # Section 6:
    #
    # normal:
    #     每张卡更新所有 parameters
    #
    # sharded:
    #     每张卡更新自己的 parameters
    #     然后 broadcast
    optimizer.step()

    return loss


# ============================================================
# Main benchmark
# ============================================================

def run(
    optimizer_type: str,
    warmup_steps: int,
    measurement_steps: int,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "optimizer_state_sharding_accounting requires CUDA GPUs."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    device = torch.device(
        f"cuda:{local_rank}"
    )

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    if world_size != 2 and rank == 0:
        print(
            f"WARNING: assignment standard configuration uses 2 GPUs, "
            f"but world_size={world_size}."
        )

    torch.manual_seed(42)

    # --------------------------------------------------------
    # 清理 memory stats
    # --------------------------------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # --------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=XL_CONFIG["d_model"],
        num_layers=XL_CONFIG["num_layers"],
        num_heads=XL_CONFIG["num_heads"],
        d_ff=XL_CONFIG["d_ff"],
        rope_theta=10_000.0,
    ).to(device)

    # 使用 Section 5 实现
    model = DDPOverlapIndividualParameters(model)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    if optimizer_type == "normal":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
        )

    elif optimizer_type == "sharded":
        optimizer = ShardedOptimizer(
            model.parameters(),
            torch.optim.AdamW,
            lr=1e-3,
        )

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}"
        )

    # --------------------------------------------------------
    # Random fixed batch
    # --------------------------------------------------------

    x = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(BATCH_SIZE, CONTEXT_LENGTH),
        device=device,
    )

    y = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(BATCH_SIZE, CONTEXT_LENGTH),
        device=device,
    )

    # ========================================================
    # Memory Accounting
    # ========================================================

    print_memory_snapshot(
        label="After model initialization",
        model=model,
        optimizer=optimizer,
        device=device,
        rank=rank,
    )

    # -------------------------
    # Forward + backward
    # -------------------------

    optimizer.zero_grad(set_to_none=True)

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
    )

    loss.backward()

    model.finish_gradient_synchronization()

    # backward 已经结束
    # activation graph 大部分已经释放
    del logits
    del loss

    torch.cuda.synchronize(device)

    print_memory_snapshot(
        label="Directly before optimizer.step()",
        model=model,
        optimizer=optimizer,
        device=device,
        rank=rank,
    )

    # -------------------------
    # 第一次 optimizer.step()
    #
    # AdamW 在这里第一次创建：
    #
    # exp_avg
    # exp_avg_sq
    # -------------------------

    optimizer.step()

    torch.cuda.synchronize(device)

    print_memory_snapshot(
        label="Directly after optimizer.step()",
        model=model,
        optimizer=optimizer,
        device=device,
        rank=rank,
    )

    # ========================================================
    # Runtime Accounting
    # ========================================================

    # 先 warmup
    for _ in range(warmup_steps):
        training_step(
            model,
            optimizer,
            x,
            y,
        )

    torch.cuda.synchronize(device)
    dist.barrier()

    times = []

    for _ in range(measurement_steps):
        torch.cuda.synchronize(device)

        start = time.perf_counter()

        training_step(
            model,
            optimizer,
            x,
            y,
        )

        torch.cuda.synchronize(device)

        end = time.perf_counter()

        times.append(end - start)

    dist.barrier()

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    if rank == 0:
        print()
        print("=" * 70)
        print("Timing Result")
        print("=" * 70)

        print(f"Optimizer:     {optimizer_type}")
        print(f"World size:    {world_size}")
        print(f"Warmup steps:  {warmup_steps}")
        print(f"Measure steps: {measurement_steps}")

        print(
            f"Iteration time: "
            f"{mean_time * 1000:.3f} "
            f"± {std_time * 1000:.3f} ms"
        )

        print()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizer",
        choices=[
            "normal",
            "sharded",
        ],
        required=True,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    run(
        optimizer_type=args.optimizer,
        warmup_steps=args.warmup,
        measurement_steps=args.steps,
    )