import math


def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """
    Linear warmup + cosine learning rate schedule.

    Args:
        t: 当前训练 step
        alpha_max: 最大学习率
        alpha_min: 最小学习率
        T_w: warmup 结束的 step
        T_c: cosine decay 结束的 step

    Returns:
        当前 step 对应的 learning rate
    """

    # 1. Warmup：学习率从 0 线性增加到 alpha_max
    if t < T_w:
        return (t / T_w) * alpha_max

    # 2. Cosine decay：从 alpha_max 平滑下降到 alpha_min
    elif t <= T_c:
        progress = (t - T_w) / (T_c - T_w)

        return alpha_min + 0.5 * (
            1 + math.cos(math.pi * progress)
        ) * (alpha_max - alpha_min)

    # 3. T_c 之后保持最小学习率
    else:
        return alpha_min