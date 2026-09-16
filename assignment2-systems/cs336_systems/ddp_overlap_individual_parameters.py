import torch
import torch.distributed as dist
import torch.nn as nn


class DDPOverlapIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()

        self.module = module
        self.world_size = dist.get_world_size()

        # 保存尚未完成的异步 AllReduce
        self.handles = []

        # 1. 初始化时，让所有 rank 的参数都和 rank 0 一致
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # 2. 给每一个需要梯度的参数注册 backward hook
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_hook()
                )

    def _make_hook(self):
        def hook(param: torch.Tensor):
            # 此时这个GPU param.grad 已经计算完成
            # 会自动更新params.grad
            # 对所有GPU的模型的同一个参数进行相加
            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )

            # 由于可能别的GPU没有完成param.backward，所以不在这里除以world_size
            # 不在这里 wait！
            # 让 backward 继续执行，从而实现通信/计算 overlap
            self.handles.append((handle, param))

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # backward 已经结束，现在 optimizer.step() 前
        # 必须确保所有 AllReduce 完成
        for handle, param in self.handles:
            handle.wait()

            # AllReduce 得到的是 SUM
            # DDP 需要平均 gradient
            # 此时所有GPU已经算完param
            param.grad.div_(self.world_size)

        # 下一轮 backward 要重新收集 handles
        self.handles.clear()