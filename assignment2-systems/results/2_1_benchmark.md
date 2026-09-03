# 2.1 Benchmarking

实验脚本：`cs336_systems/benchmarking_script.py`
测试/验证方法见：`docs/2_1_3_how_to_test.md`

固定设置（除非另行说明）：`vocab_size = 10000`，`batch_size = 4`，`context_length = 512`，warm-up = 5，measurement = 10。

## 环境 / Hardware

- GPU:
- PyTorch:
- CUDA:
- 备注（是否 OOM 等）:

## 2.1.3 (b) Benchmark Results

模型配置来自 handout Table 1。

| Model  | d_model | d_ff  | layers | heads | Forward Mean (ms) | Forward Std (ms) | Fwd+Bwd Mean (ms) | Fwd+Bwd Std (ms) | Train (w/ opt) Mean (ms) | Train Std (ms) |
|--------|--------:|------:|-------:|------:|------------------:|-----------------:|------------------:|-----------------:|-------------------------:|---------------:|
| small  | 768     | 3072  | 12     | 12    |                   |                  |                   |                  |                          |                |
| medium | 1024    | 4096  | 24     | 16    |                   |                  |                   |                  |                          |                |
| large  | 1280    | 5120  | 36     | 20    |                   |                  |                   |                  |                          |                |
| xl     | 2560    | 10240 | 32     | 32    |                   |                  |                   |                  |                          |                |
| 10B    | 4608    | 12288 | 50     | 36    |                   |                  |                   |                  |                          |                |

**回答（1–2 句）：** forward 大约 ___ ms，backward 大约 ___ ms；标准差 ___（大/小），说明测量 ___。
> TODO: 根据实验结果填写。

## 2.1.3 (c) 关于 Warm-up 的影响

| Model | warm-up 步数 | Forward Mean (ms) | Forward Std (ms) |
|-------|-------------:|------------------:|-----------------:|
| small | 0            |                   |                  |
| small | 1            |                   |                  |
| small | 2            |                   |                  |
| small | 5            |                   |                  |

**回答（2–3 句）：**
> TODO: 去掉 warm-up 后结果如何变化？为什么（一次性初始化开销：CUDA context、cuBLAS/cuDNN autotune、显存分配等）？为什么 1–2 个 warm-up 仍不够稳定？

## Observations

- TODO: forward vs backward vs train 的耗时关系是否符合预期。
- TODO: 随模型规模增大，时间如何增长。
- TODO: 是否遇到 OOM，哪个规格开始。
