import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank: int, world_size: int, backend: str, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    # 把它们组成通信组
    dist.init_process_group(
        # 典型的backend是NCCL，NVIDIA GPU 集合通信，负责通信
        backend=backend,
        rank=rank,  # 表示当前是第几个进程
        world_size=world_size,
    )


def cleanup():
    dist.destroy_process_group()


def synchronize(backend: str):
    """Wait for GPU work to finish when benchmarking NCCL."""
    if backend == "nccl":
        torch.cuda.synchronize()


def benchmark_worker(
    rank: int,
    world_size: int,
    backend: str,
    size_mb: int,
    warmup: int,
    iterations: int,
    port: int,
):
    setup(rank, world_size, backend, port)

    # ---------------------------------------------------------
    # 1. Choose device
    # ---------------------------------------------------------
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL requires CUDA GPUs.")

        if torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"Need {world_size} GPUs, "
                f"but only found {torch.cuda.device_count()}."
            )

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # ---------------------------------------------------------
    # 2. Construct float32 tensor of requested size
    #
    # float32 = 4 bytes
    # numel = total_bytes / 4
    # ---------------------------------------------------------
    # num_bytes表示字节数
    num_bytes = size_mb * 1024 * 1024
    numel = num_bytes // 4

    # 真正创建测试 tensor
    tensor = torch.empty(
        numel,
        dtype=torch.float32,
        device=device,
    )

    # ---------------------------------------------------------
    # 3. Warm-up
    #
    # NCCL especially needs warm-up before timing.
    # ---------------------------------------------------------
    for _ in range(warmup):
        # 初始化每个process的Tensor
        tensor.fill_(rank + 1)

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

        synchronize(backend)

    # Make sure every rank reaches benchmark together.
    dist.barrier()

    # ---------------------------------------------------------
    # 4. Benchmark All-Reduce
    # ---------------------------------------------------------
    times = []

    for _ in range(iterations):
        # Reset outside timed region.
        tensor.fill_(rank + 1)

        dist.barrier()

        synchronize(backend)

        start = time.perf_counter()

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

        synchronize(backend)

        end = time.perf_counter()

        times.append(end - start)

    local_avg = sum(times) / len(times)

    # ---------------------------------------------------------
    # 5. Aggregate timing results across ranks
    # ---------------------------------------------------------
    result = torch.tensor(
        [local_avg],
        dtype=torch.float64,
        device=device,
    )

    result_sum = result.clone()
    result_min = result.clone()
    result_max = result.clone()

    dist.all_reduce(result_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(result_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(result_max, op=dist.ReduceOp.MAX)

    synchronize(backend)

    avg_time = result_sum.item() / world_size
    min_time = result_min.item()
    max_time = result_max.item()

    # ---------------------------------------------------------
    # 6. Only rank 0 prints result
    # ---------------------------------------------------------
    if rank == 0:
        bandwidth_gb_s = (num_bytes / 1e9) / avg_time

        print("=" * 60)
        print("All-Reduce Benchmark")
        print(f"backend       : {backend}")
        print(f"world_size    : {world_size}")
        print(f"tensor size   : {size_mb} MB")
        print(f"iterations    : {iterations}")
        print(f"average time  : {avg_time * 1000:.3f} ms")
        print(f"min rank time : {min_time * 1000:.3f} ms")
        print(f"max rank time : {max_time * 1000:.3f} ms")
        print(f"raw bandwidth : {bandwidth_gb_s:.3f} GB/s")
        print("=" * 60)

    cleanup()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        type=str,
        choices=["gloo", "nccl"],
        default="gloo",
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--size-mb",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=29500,
    )

    args = parser.parse_args()


    # 创建 world_size 个 Python process
    mp.spawn(
        benchmark_worker,
        args=(
            args.world_size,
            args.backend,
            args.size_mb,
            args.warmup,
            args.iterations,
            args.port,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()