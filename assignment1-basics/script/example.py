from collections import Counter
import regex as re


# GPT-2 pre-tokenization regex
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# 对一个 sequence 中指定的 pair 进行合并
def merge_sequence(
    sequence: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:

    results = []
    i = 0

    while i < len(sequence):

        if (
            i < len(sequence) - 1
            and sequence[i] == pair[0]
            and sequence[i + 1] == pair[1]
        ):
            results.append(pair[0] + pair[1])
            i += 2

        else:
            results.append(sequence[i])
            i += 1

    return tuple(results)


# 根据输入文本训练 BPE
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # ------------------------------------------------
    # 1. 初始化 vocab：256 个 byte
    # ------------------------------------------------

    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    merges = []


    # ------------------------------------------------
    # 2. 加入 special tokens
    # ------------------------------------------------

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if len(vocab) > vocab_size:
        raise ValueError("vocab_size太小")


    # ------------------------------------------------
    # 3. 读取训练文本
    # ------------------------------------------------

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()


    # ------------------------------------------------
    # 4. special token 作为硬边界
    # ------------------------------------------------

    if special_tokens:

        special_pattern = "|".join(
            re.escape(token)
            for token in special_tokens
        )

        parts = re.split(
            special_pattern,
            text,
        )

    else:
        parts = [text]


    # ------------------------------------------------
    # 5. pre-tokenization
    # ------------------------------------------------

    pretoken_counts = Counter()

    for part in parts:

        for match in re.finditer(PATTERN, part):

            token = match.group()

            pretoken_counts[token] += 1


    # ------------------------------------------------
    # 6. 将 pre-token 转换为 byte sequence
    #
    # 例如：
    #
    # "low" 出现 5 次
    #
    # →
    #
    # (b'l', b'o', b'w') : 5
    # ------------------------------------------------

    token_counts = Counter()

    for token, count in pretoken_counts.items():

        encoded = token.encode("utf-8")

        sequence = tuple(
            bytes([byte])
            for byte in encoded
        )

        token_counts[sequence] += count


    # ------------------------------------------------
    # 7. 初始 pair_counts
    #
    # pair_counts[(a,b)] = (a,b) 的总出现次数
    # ------------------------------------------------

    pair_counts = Counter()


    # ------------------------------------------------
    # pair_to_sequences:
    #
    # 一个 pair 出现在哪些 sequence 中
    #
    # 例如：
    #
    # (a,b) →
    # {
    #     (a,b,c),
    #     (a,b,x)
    # }
    #
    # 用于增量更新
    # ------------------------------------------------

    pair_to_sequences = {}


    for sequence, count in token_counts.items():

        for i in range(len(sequence) - 1):

            pair = (
                sequence[i],
                sequence[i + 1],
            )

            pair_counts[pair] += count

            if pair not in pair_to_sequences:
                pair_to_sequences[pair] = set()

            pair_to_sequences[pair].add(sequence)


    # ------------------------------------------------
    # 8. BPE merge 主循环
    # ------------------------------------------------

    while len(vocab) < vocab_size:

        # 没有 pair 可以继续 merge
        if not pair_counts:
            break


        # --------------------------------------------
        # 找最高频 pair
        #
        # 如果频率相同：
        # 选择字典序最大的
        # --------------------------------------------

        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,
            ),
        )


        # 记录 merge
        merges.append(best_pair)


        # 新 token
        best_token = (
            best_pair[0]
            + best_pair[1]
        )

        vocab[len(vocab)] = best_token


        # --------------------------------------------
        # 只找到包含 best_pair 的 sequence
        # --------------------------------------------

        affected_sequences = list(
            pair_to_sequences.get(
                best_pair,
                set(),
            )
        )


        # 保存：
        #
        # sequence
        # sequence 出现次数
        # --------------------------------------------

        affected_info = [
            (
                sequence,
                token_counts[sequence],
            )
            for sequence in affected_sequences
        ]


        # --------------------------------------------
        # 9. 删除旧 sequence 对 pair_counts 的贡献
        # --------------------------------------------

        for sequence, count in affected_info:

            # 删除旧 token sequence
            del token_counts[sequence]


            # set 用来更新 pair_to_sequences
            old_pairs = set()


            for i in range(len(sequence) - 1):

                pair = (
                    sequence[i],
                    sequence[i + 1],
                )

                old_pairs.add(pair)


                # 注意：
                #
                # 如果一个 pair 在同一个 sequence
                # 中出现多次，这里会执行多次，
                # 所以 pair_counts 会正确减掉多次
                # ------------------------------------

                pair_counts[pair] -= count

                if pair_counts[pair] <= 0:
                    del pair_counts[pair]


            # ----------------------------------------
            # 从倒排索引中删除旧 sequence
            # ----------------------------------------

            for pair in old_pairs:
                # pair是pair_to_sequences的索引
                if pair in pair_to_sequences:

                    pair_to_sequences[pair].discard(
                        sequence
                    )

                    if not pair_to_sequences[pair]:

                        del pair_to_sequences[pair]


        # --------------------------------------------
        # 10. 对受到影响的 sequence 执行 merge
        # --------------------------------------------

        merged_counts = Counter()


        for sequence, count in affected_info:

            new_sequence = merge_sequence(
                sequence,
                best_pair,
            )

            merged_counts[new_sequence] += count


        # --------------------------------------------
        # 11. 把 merge 后的新 sequence 加回去
        # --------------------------------------------

        for sequence, added_count in merged_counts.items():

            token_counts[sequence] += added_count


            new_pairs = set()


            for i in range(len(sequence) - 1):

                pair = (
                    sequence[i],
                    sequence[i + 1],
                )

                new_pairs.add(pair)


                # 新增 pair 的频率
                pair_counts[pair] += added_count


            # ----------------------------------------
            # 更新 pair → sequence 索引
            # ----------------------------------------

            for pair in new_pairs:

                if pair not in pair_to_sequences:

                    pair_to_sequences[pair] = set()

                pair_to_sequences[pair].add(
                    sequence
                )


    return vocab, merges