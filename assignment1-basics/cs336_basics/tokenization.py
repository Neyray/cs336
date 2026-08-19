from __future__ import annotations
from collections.abc import Iterable,Iterator
import pickle
import regex as re

PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class Tokenization:
    def __init__(
        self,
        vocab:dict[int,bytes],
        merges:list[tuple[bytes,bytes]],
        special_tokens:list[str] | None=None
    ):
        
        self.vocab=dict(vocab)
        self.merges=list(merges)
        self.special_tokens=special_tokens or []

        self.token_to_id={
            token:id
            for id,token in self.vocab.items()
        }
        for token in self.special_tokens:
            token_bytes=token.encode("utf-8")

            if token_bytes not in self.token_to_id:
                new_id=max(self.vocab.keys(),default=-1)+1

                self.vocab[new_id]=token_bytes
                self.token_to_id[token_bytes]=new_id

        self.merge_ranks={
            pair:rank
            for rank,pair in enumerate(self.merges)
        }

        sorted_special_tokens=sorted(
            set(self.special_tokens),
            key=len,
            reverse=True
        )

        if sorted_special_tokens:
            pattern="|".join(
                re.escape(token)
                for token in sorted_special_tokens
            )

            self.special_patterns=re.compile(pattern)
        else:
            self.special_patterns=None

    @classmethod
    def from_files(
        cls,
        vocab_path:str,
        merges_path:str,
        special_tokens:list[str] | None=None
    ):
        
        with open(vocab_path,"rb")as f:
            vocab=pickle.load(f)

        with open(merges_path,"rb")as f:
            merges=pickle.load(f)
        
        return cls(vocab,merges,special_tokens)
    
    # 合并字符
    def _apply_bpe(self,pre_token:str)->list[bytes]:
        encoded=pre_token.encode("utf-8")

        tokens=[
            bytes([byte])
            for byte in encoded
        ]

        
        while len(tokens)>=2:
            best_pair=None
            best_rank=float("inf")

            for left,right in zip(tokens,tokens[1:]):
                pair=(left,right)

                pair_rank=self.merge_ranks.get(pair)

                if pair_rank is not None and pair_rank<best_rank:
                    best_pair=pair
                    best_rank=pair_rank

            if not best_pair:
                break

            # 构造新的tokens
            i=0
            new_tokens=[]

            while i<len(tokens):
                if (i<len(tokens)-1
                    and tokens[i]==best_pair[0]
                    and tokens[i+1]==best_pair[1]):
                    new_tokens.append(best_pair[0]+best_pair[1])

                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            
            tokens=new_tokens
        
        return tokens
    

    # 编码不包含特殊字符的文本
    def _encode_normal_text(self,text:str)->Iterator[int]:
        for match in PATTERN.finditer(text):
            pre_token=match.group()
            # tokens是bytes类型的列表
            tokens=self._apply_bpe(pre_token)
            for token in tokens:
                yield self.token_to_id[token]

    def _encode_text(self,text:str)->Iterator[int]:
        if self.special_patterns is None:
            yield from self._encode_normal_text(text)
            return
        
        last_end=0
        for match in self.special_patterns.finditer(text):
            # 匹配到的是特殊字符
            if match.start()>last_end:
                normal_text=text[last_end:match.start()]

                yield from self._encode_normal_text(normal_text)
            
            special_token=match.group()
            encoded=special_token.encode("utf-8")
            yield self.token_to_id[encoded]

            last_end=match.end()

        if last_end<len(text):
            yield from self._encode_normal_text(text[last_end:])

    def encode(self,text:str)->list[int]:
        return list(self._encode_text(text))
    
    def encode_iterable(self,texts:Iterable[str])->Iterator[int]:
        for text in texts:
            yield from self._encode_text(text)

    def decode(self,IDs:list[int])->str:
        byte_string=b"".join(
            self.vocab[id]
            for id in IDs
        )

        return byte_string.decode(
            encoding="utf-8",
            errors="replace",
        )