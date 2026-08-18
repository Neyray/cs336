import pickle
import time

from cs336_basics.bpe import train_bpe


INPUT_PATH = "data/owt_train.txt"
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]


start = time.time()

vocab, merges = train_bpe(
    INPUT_PATH,
    VOCAB_SIZE,
    SPECIAL_TOKENS,
)

elapsed = time.time() - start


# 保存 vocab 和 merges
with open("openwebtext_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("openwebtext_merges.pkl", "wb") as f:
    pickle.dump(merges, f)


# 找最长 token
longest_token = max(vocab.values(), key=len)


print(f"Training time: {elapsed:.2f} seconds")
print(f"Vocab size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")

print("\nLongest token:")
print(longest_token)
print(f"Length: {len(longest_token)} bytes")
print(
    "Decoded:",
    longest_token.decode("utf-8", errors="replace"),
)