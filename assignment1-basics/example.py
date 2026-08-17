from collections import Counter
import regex as re


# GPT-2 pre-tokenization regex
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# pair表示这一轮准备执行merge的两个字符
# sequence表示一个 tuple，里面有任意多个 bytes 对象
def merge_sequence(
    sequence: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:

    result = []
    i = 0

    while i < len(sequence):
        if (
            i < len(sequence) - 1
            and sequence[i] == pair[0]
            and sequence[i + 1] == pair[1]
        ):
            result.append(pair[0] + pair[1])   #如果相等，就把这两个更新为一个存到result里面
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    # result是新的tuple序列
    return tuple(result)


# vocab 保存“学到了哪些 token”（最终的词表），merges 保存“这些 token 是按什么顺序合并出来的”，也就是pair的集合
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 1. 初始化 256 个 byte token
    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    # 2. 加入 special tokens
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError("vocab_size 太小")

    merges = []

    # 3. 读取训练文本
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 4. special token 作为硬边界切开
    if special_tokens:
        special_pattern = "|".join(
            re.escape(token)
            for token in special_tokens
        )
        parts = re.split(special_pattern, text)
    else:
        parts = [text]

    # 5. pre-tokenization + 统计频率
    pretoken_counts = Counter()
    # 创建一个计数器

    for part in parts:
        # re.finditer() 会根据正则表达式 PATTERN，在文本中寻找所有匹配的位置
        for match in re.finditer(PATTERN, part):  # 这里的 match 不是字符串，而是一个正则匹配对象
            token = match.group()  # 获取这个匹配对象实际匹配到的字符串
            pretoken_counts[token] += 1

    # 6. 每个 pre-token 转换成 byte token 序列
    token_counts = Counter()

    for token, count in pretoken_counts.items():

        encoded = token.encode("utf-8")

        # 此时的encoded是数字，要转换成字符再存储到token_counts里面
        sequence = tuple(
            bytes([byte])
            for byte in encoded
        )
        # 例：token_counts[(b"l", b"o", b"w")] = 5
        token_counts[sequence] += count

    # 7. BPE merge
    while len(vocab) < vocab_size:

        pair_counts = Counter()

        # 统计所有相邻 token pair
        # 把sequence的相邻字符取出来
        for sequence, count in token_counts.items():

            for i in range(len(sequence) - 1):

                pair = (
                    sequence[i],
                    sequence[i + 1],
                )

                pair_counts[pair] += count

        # 已经没有可以合并的 pair
        if not pair_counts:
            break

        # 频率最高；
        # 如果频率相同，选择字典序最大的 pair
        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,  # pair表示字典序
            ),
        )

        # 保存 merge
        merges.append(best_pair)

        # 新 token 加入 vocab
        new_token = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = new_token

        # 对所有 pre-token 应用本轮 merge
        new_token_counts = Counter()

        for sequence, count in token_counts.items():

            new_sequence = merge_sequence(
                sequence,
                best_pair,
            )

            new_token_counts[new_sequence] += count

        token_counts = new_token_counts

    return vocab, merges