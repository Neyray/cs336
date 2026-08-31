import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    # dataset是numpy类型
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一维 token ID 数组中随机采样一个 batch。

    x: 输入 token
    y: x 中每个 token 对应的下一个 token
    x.shape(B,T)
    y.shape(B,T)
    """

    # 1. 随机选择 batch_size 个起始位置
    starting_indices = torch.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,),
    )

    # 2. 根据每个起始位置截取 context_length 个 token，作为输入
    x = torch.stack([
        # from_numpy把numpy类型转换成torch.Tensor
        torch.from_numpy(
            # astype是为了返回整数索引
            dataset[i : i + context_length].astype(np.int64)
        )
        for i in starting_indices
    ])

    # 3. 整体向右移动一个 token，作为预测目标
    y = torch.stack([
        torch.from_numpy(
            dataset[i + 1 : i + context_length + 1].astype(np.int64)
        )
        for i in starting_indices
    ])

    # 4. 移动到指定设备
    x = x.to(device)
    y = y.to(device)

    return x, y