from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.modules import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.loss import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.data_loading import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    """
    在验证集上随机取多个 batch，计算平均 loss。
    """

    # 现在不是训练，而是在评估
    model.eval()

    losses = []

    for _ in range(eval_iters):
        inputs, targets = get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        losses.append(loss.item())
    # 重新进入训练状态
    model.train()

    return sum(losses) / len(losses)


def train(args: argparse.Namespace) -> None:

    # =========================================================
    # 1. 设置随机种子和 device
    # =========================================================

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    print(f"Using device: {device}")


    # =========================================================
    # 2. 使用 np.memmap 加载训练数据
    # =========================================================

    dtype = np.dtype(args.data_dtype)

    train_data = np.memmap(
        args.train_data,
        dtype=dtype,
        mode="r",
    )

    val_data = np.memmap(
        args.val_data,
        dtype=dtype,
        mode="r",
    )

    print(f"train tokens: {len(train_data):,}")
    print(f"val tokens:   {len(val_data):,}")


    # =========================================================
    # 3. 创建 TransformerLM
    # =========================================================

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    model = model.to(device)


    # =========================================================
    # 4. 创建 AdamW optimizer
    # =========================================================

    optimizer = AdamW(
        # nn.Module内部parameters()函数，自动遍历里面注册的nn.Module,nn.Parameter
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )


    # =========================================================
    # 5. 是否从 checkpoint 恢复
    # =========================================================

    start_iteration = 0

    if args.resume is not None:
        start_iteration = load_checkpoint(
            args.resume,
            model,
            optimizer,
        )

        print(
            f"Loaded checkpoint {args.resume}, "
            f"resume from iteration {start_iteration}"
        )


    # =========================================================
    # 6. checkpoint 保存目录
    # =========================================================

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    # =========================================================
    # 7. 正式训练
    # =========================================================

    model.train()

    start_time = time.time()

    for iteration in range(start_iteration, args.max_iters):

        # -----------------------------------------------------
        # 7.1 根据当前 iteration 计算 learning rate
        # -----------------------------------------------------

        lr = get_lr_cosine_schedule(
            t=iteration,
            alpha_max=args.max_lr,
            alpha_min=args.min_lr,
            T_w=args.warmup_iters,
            T_c=args.cosine_cycle_iters,
        )

        # optimizer 里面的 lr 要更新
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


        # -----------------------------------------------------
        # 7.2 从训练数据随机拿一个 batch
        # -----------------------------------------------------

        inputs, targets = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )


        # -----------------------------------------------------
        # 7.3 清空上一轮 gradient
        # -----------------------------------------------------

        optimizer.zero_grad()


        # -----------------------------------------------------
        # 7.4 Forward
        # -----------------------------------------------------

        logits = model(inputs)


        # -----------------------------------------------------
        # 7.5 Cross Entropy
        # -----------------------------------------------------

        loss = cross_entropy(
            logits,
            targets,
        )


        # -----------------------------------------------------
        # 7.6 Backward
        # -----------------------------------------------------

        loss.backward()


        # -----------------------------------------------------
        # 7.7 Gradient clipping
        # -----------------------------------------------------

        gradient_clipping(
            model.parameters(),
            args.max_grad_norm,
        )


        # -----------------------------------------------------
        # 7.8 AdamW 更新 parameters
        # -----------------------------------------------------

        optimizer.step()


        # -----------------------------------------------------
        # 7.9 输出训练信息
        # -----------------------------------------------------

        if iteration % args.log_interval == 0:
            elapsed = time.time() - start_time

            print(
                f"iter {iteration:6d} | "
                f"train loss {loss.item():.4f} | "
                f"lr {lr:.6e} | "
                f"time {elapsed:.1f}s"
            )


        # -----------------------------------------------------
        # 7.10 validation
        # -----------------------------------------------------

        if (
            iteration % args.eval_interval == 0
            or iteration == args.max_iters - 1
        ):
            val_loss = estimate_loss(
                model=model,
                dataset=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_iters=args.eval_iters,
            )

            print(
                f">>> validation | "
                f"iter {iteration:6d} | "
                f"val loss {val_loss:.4f}"
            )


        # -----------------------------------------------------
        # 7.11 checkpoint
        # -----------------------------------------------------

        if (
            (iteration + 1) % args.checkpoint_interval == 0
            or iteration == args.max_iters - 1
        ):
            checkpoint_path = (
                checkpoint_dir
                / f"checkpoint_{iteration + 1}.pt"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration + 1,
                out=checkpoint_path,
            )

            print(f"Saved checkpoint: {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model."
    )


    # =========================================================
    # 数据
    # =========================================================

    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_dtype",
        type=str,
        default="uint16",
    )


    # =========================================================
    # 模型参数
    # =========================================================

    parser.add_argument("--vocab_size", type=int, required=True)

    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--d_ff",
        type=int,
        default=1344,
    )

    parser.add_argument(
        "--rope_theta",
        type=float,
        default=10000.0,
    )


    # =========================================================
    # optimizer 参数
    # =========================================================

    parser.add_argument(
        "--max_lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )


    # =========================================================
    # 训练参数
    # =========================================================

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max_iters",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--cosine_cycle_iters",
        type=int,
        default=10000,
    )


    # =========================================================
    # logging / validation
    # =========================================================

    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eval_interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval_iters",
        type=int,
        default=20,
    )


    # =========================================================
    # checkpoint
    # =========================================================

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )


    # =========================================================
    # 其他
    # =========================================================

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)