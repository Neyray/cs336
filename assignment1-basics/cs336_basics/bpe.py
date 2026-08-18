from collections import Counter
import regex as re


# GPT-2 pre-tokenization regex
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 根据pair来更新现有序列
def merge_sequence(
    sequence:tuple[bytes,...],
    pair:tuple[bytes,bytes],
)->tuple[bytes,...]:
    
    result=[]
    i=0

    while i < len(sequence):
        if (i<len(sequence)-1 
            and sequence[i]==pair[0]
            and sequence[i+1]==pair[1]):
            result.append(pair[0]+pair[1])
            i+=2
        else:
            result.append(sequence[i])
            i+=1
    return tuple(result)

def train_bpe(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str],
)->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    
    # 定义vocab，dict类型
    vocab={
        i:bytes([i])
        for i in range(256)
    }

    for token in special_tokens:
        vocab[len(vocab)]=token.encode("utf-8")

    if len(vocab)>vocab_size:
        raise ValueError("vocab_size太小")
    
    # 定义list类型
    merges=[]


    with open(input_path,"r",encoding="utf-8") as f:
        text=f.read()

    # 按照special_tokens来划分边界
    if special_tokens:
        special_pattern="|".join(
            re.escape(token)
            for token in special_tokens
        )

        parts=re.split(
            special_pattern,
            text
        )
    else:
        parts=[text]

    
    # 定义pretoken_counts统计单词出现次数
    pretoken_counts=Counter()
    for part in parts:
        for match in re.finditer(PATTERN,part):
            token=match.group()
            pretoken_counts[token]+=1

    

    # 定义token_counts将字母中的序列展开统计
    # dict类型
    token_counts=Counter()
    for token,count in pretoken_counts.items():
        encoded=token.encode("utf-8")
        # tuple类型
        sequence=tuple(
            bytes([byte])
            for byte in encoded
        )

        token_counts[sequence]+=count

    
    # 定义pair_counts统计相邻字符数量
    pair_counts=Counter()   # dict类型
    pair_to_sequences={}   # dict类型

    for sequence,count in token_counts.items():
        for i in range(len(sequence)-1):
            # tuple类型
            pair=(sequence[i],sequence[i+1])

            pair_counts[pair]+=count

            if pair not in pair_to_sequences:
                pair_to_sequences[pair]=set()
            
            pair_to_sequences[pair].add(sequence)

    # merge主循环
    while len(vocab)<vocab_size:
        if not pair_counts:
            break

        # tuple类型
        best_pair=max(
            pair_counts,
            key=lambda pair:(
                pair_counts[pair],
                pair,
            ),
        )

        #补充merges
        merges.append(best_pair)

        best_tokens=(best_pair[0]+best_pair[1])
        vocab[len(vocab)]=best_tokens





        # 列表类型
        affected_sequences=list(
            pair_to_sequences.get(
                best_pair,
                set(),
            )
        )

        # list类型，内部存储tuple
        affected_info=[
            (sequence,token_counts[sequence])
            for sequence in affected_sequences
        ]



        for sequence,count in affected_info:
            # 1.删除token_counts中受影响的序列
            del token_counts[sequence]

            old_pairs=set()  # 存储受到影响的元组（相邻元素）

            for i in range(len(sequence)-1):
                pair=(sequence[i],sequence[i+1])
                old_pairs.add(pair)

                # 2.删除pair_counts中受到影响的sequence中的相邻元组
                pair_counts[pair]-=count

                if pair_counts[pair]<=0:
                    del pair_counts[pair]

            # 3.删除pair_to_sequences中pair关于受到影响的sequence索引
            for pair in old_pairs:
                if pair in pair_to_sequences:
                    pair_to_sequences[pair].discard(sequence)
                
                if not pair_to_sequences[pair]:
                    del pair_to_sequences[pair]

        merge_counts=Counter()  # dict类型
        for sequence,count in affected_info:
            new_sequence=merge_sequence(sequence,best_pair)

            merge_counts[new_sequence]+=count
        

        # 更新数据
        for sequence,count in merge_counts.items():
            # 1.更新token_counts
            token_counts[sequence]+=count
            # 原先的删去了，现在需要加入新的

            new_pairs=set()

            for i in range(len(sequence)-1):
                pair=(sequence[i],sequence[i+1])

                new_pairs.add(pair)

                # 2.更新pair_counts
                pair_counts[pair]+=count

            
            # 3.更新pair_to_sequences
            for pair in new_pairs:
                if pair not in pair_to_sequences:
                    pair_to_sequences[pair]=set()
                pair_to_sequences[pair].add(sequence)
    return vocab,merges