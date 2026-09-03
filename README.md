# CS336：从零开始构建语言模型

这是我的 Stanford CS336 学习与作业实现仓库。

整个仓库采用单仓库（monorepo）结构管理，不同 Assignment 作为根目录下的普通子文件夹统一维护。

## 仓库结构

```text
ModelForge/
├── assignment1-basics/
├── assignment2-systems/
└── README.md
```

## Assignment 1：Basics

`assignment1-basics/` 主要实现从零构建和训练 Transformer Language Model 所需要的核心模块。

主要内容包括：

- BPE Tokenizer
- Linear
- Embedding
- RMSNorm
- SwiGLU
- RoPE
- Scaled Dot-Product Attention
- Multi-Head Self-Attention
- Transformer Block
- Transformer Language Model
- Cross Entropy
- AdamW
- Learning Rate Scheduler
- Gradient Clipping
- Data Loading
- Checkpointing
- Decoding
- Text Generation

目录：

```text
assignment1-basics/
```

## Assignment 2：Systems

`assignment2-systems/` 主要学习和实现大语言模型训练中的系统相关内容。

目录：

```text
assignment2-systems/
```

后续会随着学习进度继续补充具体实现内容。

## 开发环境

主要开发环境：

- Windows 11 + WSL Ubuntu
- Python
- PyTorch
- Conda
- uv
- pytest

激活环境：

```bash
conda activate cs336
```

运行全部测试：

```bash
uv run pytest
```

运行指定测试：

```bash
uv run pytest -k test_name
```

## Git 管理方式

整个 CS336 项目只使用一个 Git 仓库。

```text
ModelForge/
├── .git/
├── assignment1-basics/
├── assignment2-systems/
└── README.md
```

`ModelForge/.git` 是整个项目唯一的 Git 仓库。

各个 Assignment 目录都作为普通子目录管理，不保留自己的 `.git`。

后续新增 Assignment 时，也会继续按照相同方式放在 `ModelForge/` 根目录下统一管理。