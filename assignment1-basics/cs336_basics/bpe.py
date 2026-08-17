from collections import Counter
import regex as re

# GPT-2 pre-tokenization regex
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 对一组序列进行合并，更新序列
def merge_sequence(
    sequence:tuple[bytes,...],
    pair:tuple[bytes,bytes],
)->tuple[bytes,...]:
    results=[]
    i=0

    while i<len(sequence):
        if(i<len(sequence)-1
           and sequence[i]==pair[0]
           and sequence[i+1]==pair[1]):
            results.append(pair[0]+pair[1])
            i+=2
        else:
            results.append(sequence[i])
            i+=1

    return tuple(results)

# 根据输入文段构造BPE
def train_bpe(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str]
)->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:  # bytes是什么数据类型
    vocab={  # 对vocab进行初始化
        i:bytes([i]) 
        for i in range(256)
    }   # 什么时候用{}()

    merges=[]

    # 加入special_tokens
    for token in special_tokens:
        vocab[len(vocab)]=token.encode("utf-8")
    
    if len(vocab)>vocab_size:
        raise ValueError("vocab_size太小")
    
    # 读取训练文本
    with open(input_path,"r",encoding="utf-8") as f:
        text=f.read()

    if special_tokens:  # 如果special_tokens存在，就要把训练文本按照special_tokens切开
        special_pattern="|".join(
            re.escape(tokens)
            for tokens in special_tokens
        )
        parts=re.split(special_pattern,text)
    else:
        parts=[text]  # 没有special_tokens，就不需要切分
    # 对每部分part进行单词计数
    pretoken_counts=Counter()

    for part in parts:
        for match in re.finditer(PATTERN,part):
            token=match.group() # 还原为真实字符串
            pretoken_counts[token]+=1

    # 根据现有的单词个数构造字符数目，拆开单词
    token_counts=Counter()
    for token,count in pretoken_counts.items():
        encoded=token.encode("utf-8")

        sequence=tuple(
            bytes([byte])
            for byte in encoded
        )
        token_counts[sequence]+=count  # 例：token_counts[(b"l", b"o", b"w")] = 5
    
    # 主循环，进行BPE merge
    while len(vocab)<vocab_size:
        pair_counts=Counter()

        # 构造相邻字符数目
        for sequence,count in token_counts.items():
            for i in range(len(sequence)-1):
                pair=(sequence[i],
                      sequence[i+1])
                pair_counts[pair]+=count

        if not pair_counts:
            break
        best_pair=max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,
            ),
        )

        merges.append(best_pair)

        best_tokens=best_pair[0]+best_pair[1]
        vocab[len(vocab)]=best_tokens

        # 更新token_counts
        new_token_counts=Counter()

        for sequence,count in token_counts.items():
            new_sequence=merge_sequence(
                sequence,
                best_pair
            )
            new_token_counts[new_sequence]+=count
            # 把新的匹配的字符更新到序列里

        token_counts=new_token_counts
    return vocab,merges