from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    Optimizer State Sharding.

    每个 rank：
        - 拥有完整 parameters
        - 拥有完整 gradients
        - 只维护一部分 parameters 的 optimizer states

    optimizer.step() 后：
        - 每个 rank 更新自己负责的参数
        - 然后 broadcast 给其他 rank
        - 最终所有 rank 的模型参数重新保持一致
    """

    def __init__(
        self,
        params,
        optimizer_cls: type[Optimizer],
        **kwargs: Any,
    ):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "ShardedOptimizer requires an initialized distributed process group."
            )

        self.optimizer_cls = optimizer_cls

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 用来决定：
        #
        # param0 -> rank0
        # param1 -> rank1
        # param2 -> rank0
        # param3 -> rank1
        #
        # 对 world_size = 2 来说。
        self._next_param_index = 0

        # id(parameter) -> owner rank
        self._param_owner: dict[int, int] = {}

        # super().__init__ 会调用我们的 add_param_group()
        # 此时 local optimizer 还没有创建，
        # 所以先把 local param groups 暂存在这里。
        self._pending_local_groups: list[dict[str, Any]] = []

        self._local_optimizer: Optimizer | None = None

        # 注意：
        # 这里必须把“全部参数”交给 Optimizer 父类。
        #
        # 原因：
        # sharded_optimizer.zero_grad()
        # 必须能够清除所有参数的梯度，而不仅仅是本 rank 负责的参数。
        super().__init__(params, kwargs)

        # 每个 rank 真正执行参数更新的 optimizer
        #
        # 它只包含当前 rank 负责的参数。
        self._local_optimizer = optimizer_cls(
            self._pending_local_groups,
            **kwargs,
        )

        self._pending_local_groups = []

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """
        添加 parameter group，并把其中的参数分配给不同 rank。

        例如 world_size = 2：

        p0 -> rank0
        p1 -> rank1
        p2 -> rank0
        p3 -> rank1
        """

        # 先让 PyTorch Optimizer 正常维护完整 param_groups
        super().add_param_group(param_group)

        # super().add_param_group() 添加后的 group
        full_group = self.param_groups[-1]

        local_params = []

        for param in full_group["params"]:
            param_id = id(param)

            # 第一次看到这个 Parameter，给它分配 owner
            if param_id not in self._param_owner:
                owner_rank = self._next_param_index % self.world_size

                self._param_owner[param_id] = owner_rank
                self._next_param_index += 1

            owner_rank = self._param_owner[param_id]

            if owner_rank == self.rank:
                local_params.append(param)

        # 保留这个 parameter group 的超参数：
        #
        # lr
        # weight_decay
        # betas
        # eps
        # ...
        #
        # 只是把 params 换成本 rank 的参数。
        local_group = {
            key: value
            for key, value in full_group.items()
            if key != "params"
        }

        local_group["params"] = local_params

        # __init__ 阶段：
        # local optimizer 还没创建
        if self._local_optimizer is None:
            self._pending_local_groups.append(local_group)

        # 训练过程中动态 add_param_group
        else:
            self._local_optimizer.add_param_group(local_group)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """
        1. 当前 rank 只更新自己拥有的 parameters
        2. 每个 owner 把更新后的 parameter broadcast 给其他 rank
        """

        assert self._local_optimizer is not None

        # 如果外面通过 scheduler 或者直接修改：
        #
        # optimizer.param_groups[i]["lr"]
        #
        # 要同步给真正的 local optimizer。
        for full_group, local_group in zip(
            self.param_groups,
            self._local_optimizer.param_groups,
        ):
            for key, value in full_group.items():
                if key != "params":
                    local_group[key] = value

        # -------------------------
        # 1. 更新 local parameters
        # -------------------------

        if closure is None:
            loss = self._local_optimizer.step(**kwargs)
        else:
            loss = self._local_optimizer.step(
                closure=closure,
                **kwargs,
            )

        # -------------------------
        # 2. 同步更新后的 parameters
        # -------------------------
        #
        # 比如：
        #
        # p0 owner = rank0
        # p1 owner = rank1
        #
        # rank0:
        #     p0_new
        #     p1_old
        #
        # rank1:
        #     p0_old
        #     p1_new
        #
        # broadcast 后：
        #
        # rank0:
        #     p0_new
        #     p1_new
        #
        # rank1:
        #     p0_new
        #     p1_new

        for group in self.param_groups:
            for param in group["params"]:
                owner_rank = self._param_owner[id(param)]

                dist.broadcast(
                    param.data,
                    src=owner_rank,
                )

        return loss

    @property
    def local_optimizer(self) -> Optimizer:
        """
        Task 2 做 memory accounting 时会用到。
        """
        assert self._local_optimizer is not None
        return self._local_optimizer