from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cs336_basics.tokenization import Tokenization


SPECIAL_TOKENS = ["<|endoftext|>"]


def encode_dataset(
    input_path: str,
    vocab_path: str,
    merges_path: str,
    output_path: str,
    dtype: str = "uint16",
    chunk_size: int = 1_000_000,
) -> None:

    # 1. 加载你已经训练好的 tokenizer
    tokenizer = Tokenization.from_files(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=SPECIAL_TOKENS,
    )

    np_dtype = np.dtype(dtype)

    # 确保 token id 能装进 uint16
    max_token_id = max(tokenizer.vocab.keys())

    if max_token_id > np.iinfo(np_dtype).max:
        raise ValueError(
            f"Token id {max_token_id} cannot fit in {dtype}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    buffer = []
    total_tokens = 0

    # 2. 流式读取文本
    with open(input_path, "r", encoding="utf-8") as f, \
         open(output_path, "wb") as out:

        # 直接调用你已经实现好的 encode_iterable
        for token_id in tokenizer.encode_iterable(f):

            buffer.append(token_id)

            # 避免整个 OWT token 全塞进内存
            if len(buffer) >= chunk_size:
                arr = np.asarray(
                    buffer,
                    dtype=np_dtype,
                )

                arr.tofile(out)

                total_tokens += len(buffer)
                buffer.clear()

        # 写最后剩下的一部分
        if buffer:
            arr = np.asarray(
                buffer,
                dtype=np_dtype,
            )

            arr.tofile(out)

            total_tokens += len(buffer)

    print(f"Finished: {input_path}")
    print(f"Output:   {output_path}")
    print(f"Tokens:   {total_tokens:,}")
    print(f"Dtype:    {dtype}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--merges",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1_000_000,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    encode_dataset(
        input_path=args.input,
        vocab_path=args.vocab,
        merges_path=args.merges,
        output_path=args.output,
        dtype=args.dtype,
        chunk_size=args.chunk_size,
    )