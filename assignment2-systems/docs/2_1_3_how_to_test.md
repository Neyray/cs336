# 2.1.3 End-to-End Benchmarking —— 怎么测试

> 对应题目：**Problem (benchmarking_script)**（4 points）
> 真正的实现文件放在 `cs336_systems/benchmarking_script.py`（`script/` 里的是练习文件，不交）。
> 结果和文字回答填到 `results/2_1_benchmark.md`。

本文件只讲**怎么验证 / 怎么跑 / 要注意什么**，不含实现代码——代码是你自己要写的部分。

---

## 一、题目要你做什么（拆成 a / b / c）

**(a) 写脚本**，支持：
- 给定超参数（层数等）初始化模型；
- 造一个随机 batch 的数据（随机权重、随机数据都行，因为只测速度和显存）；
- 跑 `w` 个 warm-up 步（不计时），再对 `n` 个步计时；模式由命令行参数控制：
  - 只 forward
  - forward + backward
  - forward + backward + optimizer.step
- 每步之后调用 `torch.cuda.synchronize()`。

**(b)** 用 2.1.2 的 5 个模型规格跑一遍，**5 warm-up + 10 measurement**，报告 mean/std，回答：forward 多久？backward 多久？方差大不大？（1–2 句）

**(c)** 去掉 warm-up 再跑一遍，对比结果、解释原因；再试 1–2 个 warm-up 步，说明为什么还是不一样。（2–3 句）

---

## 二、跑之前先自查（sanity check）

1. **解释器 / 依赖对不对**
   ```bash
   uv run python -c "import cs336_basics.model; print('ok')"
   ```
   能打印 `ok` 说明模型包能导入（见 2.1.1）。

2. **模型配置要对齐 Table 1（这是最容易错的地方）**
   handout 里 2.1.2 的规格是：

   | Size | d_model | d_ff | num_layers | num_heads |
   |------|--------:|-----:|-----------:|----------:|
   | small | 768 | 3072 | 12 | 12 |
   | medium | 1024 | 4096 | 24 | 16 |
   | large | 1280 | 5120 | 36 | 20 |
   | xl | 2560 | 10240 | 32 | 32 |
   | 10B | 4608 | 12288 | 50 | 36 |

   > ⚠️ 注意：`script/example.py` 里的 `xl` 配置（1600/6400/48/25）和 handout **不一致**，也没有 `10B`。正式脚本要以上表为准。
   - `vocab_size = 10000`，`batch_size = 4`，**context length 默认 512**（除非题目另有说明）。

3. **d_model 能被 num_heads 整除**（每个规格都满足，改配置时留意）。

---

## 三、怎么手动测试脚本是对的

按"由小到大、由易到难"的顺序验证，别一上来就跑 xl/10B：

1. **先用超小配置冒烟测试**，确认三种 mode 都不报错、都能打印 mean/std：
   ```bash
   uv run python cs336_systems/benchmarking_script.py --model-size small --mode forward         --warmup-steps 1 --measurement-steps 2 --context-length 32 --batch-size 2
   uv run python cs336_systems/benchmarking_script.py --model-size small --mode forward_backward --warmup-steps 1 --measurement-steps 2 --context-length 32 --batch-size 2
   uv run python cs336_systems/benchmarking_script.py --model-size small --mode train           --warmup-steps 1 --measurement-steps 2 --context-length 32 --batch-size 2
   ```
   （参数名以你自己脚本里的 argparse 为准，这里只是示意。）

2. **验证三种 mode 的耗时关系符合直觉**：
   `forward < forward_backward < train`
   如果 backward 反而比 forward 快，多半是计时/同步写错了。

3. **验证 CUDA 同步真的生效（只有 GPU 上才有意义）**：
   - 有 GPU 时，去掉 `torch.cuda.synchronize()` 再测，如果时间**突然变得极小且方差极大**，说明你之前测到的其实是"CPU 把 kernel 丢给 GPU 就返回"的时间，不是真正的计算时间。加回同步后数字才可信。
   - CPU 上 `synchronize` 是 no-op，无法验证这一点——这题的重点本来就是 GPU。

4. **验证 warm-up 的作用（这就是 (c) 要回答的）**：
   - `--warmup-steps 0` vs `5`，看第一次测量是不是明显偏慢、std 是不是明显变大。
   - 原因方向（自己想清楚再写）：首次运行有 CUDA context 初始化、cuDNN/cuBLAS 算法选择（autotune）、显存分配、`torch.compile`/JIT 等一次性开销；1–2 个 warm-up 未必把这些全部"预热"完。

---

## 四、正式测量（回答 b）

- 固定 `--warmup-steps 5 --measurement-steps 10`，对 small→10B 各跑一遍；
- 每个规格分别测 forward-only 和 forward+backward（optimizer step 也可单独记）；
- 把 mean/std 填进 `results/2_1_benchmark.md` 的表格；
- 显存/算力不够时，大模型（xl / 10B）可能 OOM——这本身也是一个可以记录和讨论的结果，不要硬跑到崩。

> handout 建议用代码自动生成表格（`pandas.DataFrame.to_latex()` / `to_typst()`），省得手抄。

---

## 五、常见坑

- 计时用 `timeit.default_timer()`，不要用 `time.time()`。
- 计时区间**不要**把随机数据生成、模型构造算进去——只计时那 `n` 个步。
- `train` 模式记得每步 `optimizer.zero_grad()`，否则梯度累积会影响结果。
- 报告时写清硬件（GPU 型号、PyTorch/CUDA 版本），否则数字没法比较。
