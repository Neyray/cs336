from __future__ import annotations  # 让 Python 延迟处理类型注解

from collections.abc import Iterable, Iterator
import pickle  # 把 Python 对象保存到文件，再从文件恢复

import regex as re


# 和训练 BPE 时使用同一个 GPT-2 pre-tokenization pattern
# 把正则表达式字符串编译成一个“正则对象”
PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        # bytes -> token id
        self.token_to_id = {
            token: token_id
            for token_id, token in self.vocab.items()
        }

        # 如果 special token 不在 vocab 中，就加入 vocab
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")

            if token_bytes not in self.token_to_id:
                new_id = max(self.vocab.keys(), default=-1) + 1

                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id

        # merge pair -> merge 优先级
        # rank 越小，说明越早学到，优先级越高
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        # special token 要优先匹配最长的
        sorted_special_tokens = sorted(
            set(self.special_tokens),
            key=len,
            reverse=True,
        )

        if sorted_special_tokens:
            pattern = "|".join(
                # 把字符串里的正则特殊字符转义，让它变成普通字符
                re.escape(token)
                for token in sorted_special_tokens
            )
            self.special_pattern = re.compile(pattern)   # compile转换成标准正则
        else:
            self.special_pattern = None




    @classmethod  # 不用创建对象，直接通过类调用的方法
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        这里采用 pickle 保存 vocab / merges。
        因此你的 BPE 训练代码如果要保存文件，也要使用 pickle。
        """

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)   # 等价于return Tokenizer(vocab, merges, special_tokens)




    def _apply_bpe(self, pretoken: str) -> list[bytes]:
        """
        对一个 pre-token 应用已经训练好的 BPE merges。
        """

        encoded = pretoken.encode("utf-8")

        tokens = [
            bytes([byte])
            for byte in encoded
        ]

        while len(tokens) >= 2:

            best_pair = None
            best_rank = float("inf")

            # 找当前 token 序列中优先级最高的 pair
            for left, right in zip(tokens, tokens[1:]):
                pair = (left, right)

                rank = self.merge_ranks.get(pair)

                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            # 没有 pair 可以继续 merge
            if best_pair is None:
                break

            # merge 所有 best_pair
            new_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(
                        tokens[i] + tokens[i + 1]
                    )
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens




    def _encode_normal_text(self, text: str) -> Iterator[int]:
        """
        编码不包含 special token 的普通文本。
        """
        # PATTERN是正则对象，直接finditer
        for match in PATTERN.finditer(text):

            # 每个单词
            pretoken = match.group()
            # merge后的单词
            bpe_tokens = self._apply_bpe(pretoken)

            for token in bpe_tokens:
                yield self.token_to_id[token]   # 需要一个，就产生一个




    def _encode_text(self, text: str) -> Iterator[int]:

        # 没有 special token，直接正常编码
        if self.special_pattern is None:
            yield from self._encode_normal_text(text)
            return

        last_end = 0

        # 找 special tokens
        # 按照special_pattern来划分text
        for match in self.special_pattern.finditer(text):

            # special token 前面的普通文本
            if match.start() > last_end:
                normal_text = text[last_end:match.start()]
                yield from self._encode_normal_text(normal_text)
            
            # 整体直接查 ID，special token 不能再被拆成 bytes 做 BPE
            special_token = match.group()
            special_bytes = special_token.encode("utf-8")

            yield self.token_to_id[special_bytes]

            last_end = match.end()

        # 最后剩下的普通文本
        if last_end < len(text):
            yield from self._encode_normal_text(text[last_end:])




    def encode(self, text: str) -> list[int]:
        return list(self._encode_text(text))



    # 适合读大文件
    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:

        for text in iterable:
            yield from self._encode_text(text)




    def decode(self, ids: list[int]) -> str:

        byte_string = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return byte_string.decode(
            "utf-8",
            errors="replace",
        )