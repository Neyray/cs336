from __future__ import annotations

import argparse

import torch

from cs336_basics.modules import TransformerLM
from cs336_basics.tokenization import Tokenization
from cs336_basics.decoding import decode


SPECIAL_TOKENS = ["<|endoftext|>"]


def load_model(args: argparse.Namespace, device: str) -> TransformerLM:
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    # 只加载模型权重，不需要 optimizer（生成阶段用不到）
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    model = model.to(device)
    model.eval()

    return model


def main() -> None:
    args = parse_args()

    device = args.device
    print(f"Using device: {device}")

    # 1. 加载训练好的 BPE tokenizer
    tokenizer = Tokenization.from_files(
        vocab_path=args.vocab,
        merges_path=args.merges,
        special_tokens=SPECIAL_TOKENS,
    )

    # 2. 构建模型并载入 checkpoint 权重（配置必须与训练时一致）
    model = load_model(args, device)

    # 3. 采样生成
    text = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("=" * 60)
    print(text)
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Transformer LM."
    )

    # tokenizer / checkpoint
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--merges", type=str, required=True)

    # 模型参数（必须和训练时一致）
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # 采样参数
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
