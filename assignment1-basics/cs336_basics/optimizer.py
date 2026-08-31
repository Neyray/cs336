import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    # 只负责改参数，不记录梯度，不成为计算图的一部分被autograd追踪
    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 一个 optimizer 可以有多个 parameter group
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 遍历模型中的每一个 Parameter
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # self.state[p]：
                # 专门保存参数 p 对应的 AdamW 历史状态，相当于字典
                state = self.state[p]

                # 第一次遇到这个参数时初始化
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                # t 从 1 开始
                state["step"] += 1
                t = state["step"]

                m = state["m"]
                v = state["v"]

                # 1. Weight Decay
                p.mul_(1 - lr * weight_decay)

                # 2. 更新一阶矩 m
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 3. 更新二阶矩 v
                # value 只是一个矩阵缩放系数
                v.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # 4. bias correction 合并到学习率中
                adjusted_lr = (
                    lr
                    * math.sqrt(1 - beta2**t)
                    / (1 - beta1**t)
                )

                # 5. Adam 参数更新
                denominator = torch.sqrt(v) + eps

                p.addcdiv_(
                    m,
                    denominator,
                    value=-adjusted_lr,
                )

        return loss