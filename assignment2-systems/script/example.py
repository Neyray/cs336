import argparse
import csv
import gc
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import (
    BasicsTransformerLM,
    RMSNorm,
    RotaryEmbedding,
    TransformerBlock,
)


XL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 2048,
    "d_model": 2560,
    "num_layers": 32,
    "num_heads": 32,
    "d_ff": 10_240,
    "rope_theta": 10_000.0,
}


def mib(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


class SavedTensorStats:
    """Count tensors that autograd saves for backward."""

    def __init__(self, *, skip_parameters: bool, verbose: bool = True):
        self.skip_parameters = skip_parameters
        self.verbose = verbose
        self.total_size_bytes = 0

    def pack_hook(self, tensor: torch.Tensor):
        if self.skip_parameters and isinstance(tensor, nn.Parameter):
            return tensor

        self.total_size_bytes += tensor.numel() * tensor.element_size()
        if self.verbose:
            print(
                "Saving residual: "
                f"shape={tensor.shape}, dtype={tensor.dtype}, grad_fn={tensor.grad_fn}"
            )
        return tensor

    def unpack_hook(self, tensor: torch.Tensor):
        if self.verbose:
            print(
                "Loading residual: "
                f"shape={tensor.shape}, dtype={tensor.dtype}, grad_fn={tensor.grad_fn}"
            )
        return tensor


def rmsnorm_residual_demo(device: torch.device) -> None:
    """Section 3.1: compare eager RMSNorm and torch.compile fused RMSNorm."""
    x = torch.randn((4, 512, 2560), device=device, requires_grad=True)

    def run(module: nn.Module, title: str) -> None:
        module.zero_grad(set_to_none=True)
        x.grad = None
        stats = SavedTensorStats(skip_parameters=False, verbose=True)

        print(f"\n===== {title} =====")
        with torch.autograd.graph.saved_tensors_hooks(
            stats.pack_hook, stats.unpack_hook
        ):
            y = module(x)
            y.sum().backward()

        print(f"Saved tensors: {mib(stats.total_size_bytes):.2f} MiB")

    eager_ln = RMSNorm(x.shape[-1], device=device)
    run(eager_ln, "RMSNorm (eager)")

    compiled_ln = torch.compile(RMSNorm(x.shape[-1], device=device))
    run(compiled_ln, "RMSNorm (torch.compile)")


def block_residual_demo(device: torch.device) -> None:
    """Section 3.2 handout demo: saved tensors for one compiled XL-sized block.

    The handout's standalone block example uses num_heads=16 even though the
    Table-1 XL model uses num_heads=32. We preserve 16 here so the experiment
    matches the handout example; the full-model sweep below uses the actual
    Table-1 XL configuration (32 heads).
    """
    if device.type != "cuda":
        raise RuntimeError("The XL TransformerBlock residual demo should be run on CUDA.")

    batch_size = 4
    context_length = 2048
    d_model = 2560
    d_ff = 10240
    num_heads = 16

    rope = RotaryEmbedding(
        context_length=context_length,
        dim=d_model // num_heads,
        theta=10_000.0,
    )
    block = TransformerBlock(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        positional_encoder=rope,
    ).to(device)

    block = torch.compile(block, fullgraph=True)
    x = torch.randn(
        (batch_size, context_length, d_model),
        device=device,
        requires_grad=True,
    )

    stats = SavedTensorStats(skip_parameters=True, verbose=False)
    with torch.autograd.graph.saved_tensors_hooks(
        stats.pack_hook, stats.unpack_hook
    ):
        _ = block(x)

    torch.cuda.synchronize(device)
    print(
        "Total size of saved tensors in single TransformerBlock: "
        f"{mib(stats.total_size_bytes):.2f} MiB"
    )


def run_layer_range(
    layers: nn.ModuleList,
    start: int,
    end: int,
    x: torch.Tensor,
) -> torch.Tensor:
    for i in range(start, end):
        x = layers[i](x)
    return x


def forward_with_checkpointing(
    model: BasicsTransformerLM,
    token_ids: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Run the LM while checkpointing consecutive Transformer blocks.

    group_size=1  -> checkpoint each TransformerBlock separately
    group_size=2  -> checkpoint every two TransformerBlocks
    group_size=4  -> checkpoint every four TransformerBlocks
    group_size=0  -> no checkpointing (baseline; likely OOM for XL/2048)
    """
    x = model.token_embeddings(token_ids)

    if group_size == 0:
        for layer in model.layers:
            x = layer(x)
    else:
        if group_size < 1:
            raise ValueError("group_size must be >= 1, or 0 for the baseline")

        n_layers = len(model.layers)
        for start in range(0, n_layers, group_size):
            end = min(start + group_size, n_layers)

            # Bind start/end now; otherwise Python's closure would see the
            # loop variables after they changed.
            def segment(inp: torch.Tensor, start=start, end=end):
                return run_layer_range(model.layers, start, end, inp)

            x = checkpoint(segment, x, use_reentrant=False)

    x = model.ln_final(x)
    return model.lm_head(x)


def recursive_checkpoint_forward(
    layers: nn.ModuleList,
    x: torch.Tensor,
    start: int = 0,
    end: int | None = None,
) -> torch.Tensor:
    """Balanced recursive checkpointing sketch for problem 3(a).

    This is included as the code sketch for the asymptotic argument. The
    measured experiment in part (b) intentionally does NOT call this function,
    because part (b) forbids nested checkpoint calls.
    """
    if end is None:
        end = len(layers)

    if end - start == 1:
        return layers[start](x)

    mid = (start + end) // 2

    def left(inp: torch.Tensor):
        return recursive_checkpoint_forward(layers, inp, start, mid)

    def right(inp: torch.Tensor):
        return recursive_checkpoint_forward(layers, inp, mid, end)

    x = checkpoint(left, x, use_reentrant=False)
    x = checkpoint(right, x, use_reentrant=False)
    return x


def one_backward_step(
    model: BasicsTransformerLM,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    group_size: int,
) -> tuple[float, float, float]:
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(token_ids.device)

    baseline_bytes = torch.cuda.memory_allocated(token_ids.device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    logits = forward_with_checkpointing(model, token_ids, group_size)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    loss.backward()
    end.record()

    torch.cuda.synchronize(token_ids.device)

    peak_bytes = torch.cuda.max_memory_allocated(token_ids.device)
    elapsed_ms = start.elapsed_time(end)

    return gib(baseline_bytes), gib(peak_bytes), elapsed_ms


def checkpoint_sweep(
    group_sizes: list[int],
    output_csv: str | None,
    batch_size: int = 4,
) -> None:
    """Problem 3(b): sweep non-nested checkpoint segment sizes on XL/2048."""
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint memory profiling requires a CUDA GPU.")

    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    print("Creating Table-1 XL model:")
    print(XL_CONFIG)
    model = BasicsTransformerLM(**XL_CONFIG).to(device)
    model.train()

    token_ids = torch.randint(
        0,
        XL_CONFIG["vocab_size"],
        (batch_size, XL_CONFIG["context_length"]),
        device=device,
    )
    targets = torch.randint(
        0,
        XL_CONFIG["vocab_size"],
        (batch_size, XL_CONFIG["context_length"]),
        device=device,
    )

    rows: list[dict[str, float | int | str]] = []

    print("\ncheckpoint sweep")
    print("group_size | checkpoints | baseline_GiB | peak_GiB | step_ms")
    print("-----------+-------------+--------------+----------+--------")

    for group_size in group_sizes:
        if group_size < 0:
            raise ValueError("group sizes must be >= 0")

        checkpoints = (
            0
            if group_size == 0
            else math.ceil(XL_CONFIG["num_layers"] / group_size)
        )

        try:
            baseline_gib, peak_gib, elapsed_ms = one_backward_step(
                model=model,
                token_ids=token_ids,
                targets=targets,
                group_size=group_size,
            )
            status = "ok"
            print(
                f"{group_size:10d} | {checkpoints:11d} | "
                f"{baseline_gib:12.3f} | {peak_gib:8.3f} | {elapsed_ms:7.1f}"
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            baseline_gib = float("nan")
            peak_gib = float("nan")
            elapsed_ms = float("nan")
            status = "OOM"
            print(f"{group_size:10d} | {checkpoints:11d} |          OOM")

        rows.append(
            {
                "group_size": group_size,
                "num_checkpoints": checkpoints,
                "baseline_gib": baseline_gib,
                "peak_gib": peak_gib,
                "step_ms": elapsed_ms,
                "status": status,
            }
        )

        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    valid_rows = [row for row in rows if row["status"] == "ok"]
    if valid_rows:
        best = min(valid_rows, key=lambda row: float(row["peak_gib"]))
        print(
            "\nMeasured minimum: "
            f"group_size={best['group_size']}, peak={float(best['peak_gib']):.3f} GiB"
        )

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["rmsnorm", "block", "checkpoint"],
        required=True,
    )
    parser.add_argument(
        "--group-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
        help="Consecutive blocks per non-nested checkpoint. Use 0 for baseline.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/gradient_checkpointing.csv",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.experiment == "rmsnorm":
        rmsnorm_residual_demo(device)
    elif args.experiment == "block":
        block_residual_demo(device)
    else:
        checkpoint_sweep(
            group_sizes=args.group_sizes,
            output_csv=args.output_csv,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
