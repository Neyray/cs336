import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed process group must be initialized before DDP"
            )

        self.module = module
        self.world_size = dist.get_world_size()

        # 保存异步 all_reduce 返回的 handles
        self._handles = []

        # -------------------------------------------------
        # 1. 初始化：把 rank 0 的参数广播给所有 rank
        # -------------------------------------------------
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

        # -------------------------------------------------
        # 2. 为每个需要梯度的参数注册 backward hook
        #
        # 当 param.grad 准备好后：
        #     立刻启动 asynchronous all_reduce
        #
        # 从而让：
        #     backward computation
        #     communication
        # 尽可能 overlap
        # -------------------------------------------------
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_grad_hook(param)
                )

    def _make_grad_hook(self, param: torch.Tensor):
        def hook(_):
            if param.grad is None:
                return

            # local gradient / world_size
            #
            # 然后 all_reduce SUM：
            #
            #   g = (g0 + g1 + ... + gn) / world_size
            #
            param.grad.div_(self.world_size)

            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )

            self._handles.append(handle)

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        在 backward 完成之后、optimizer.step() 之前调用。

        确保所有 asynchronous all_reduce 已经完成。
        """
        for handle in self._handles:
            handle.wait()

        self._handles.clear()