from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class ShardedParameter:
    param: nn.Parameter

    original_shape: torch.Size
    original_numel: int

    shard_numel: int

    # 当前 rank 真正长期拥有的参数
    local_shard: torch.Tensor


class FullyShardedDataParallel(nn.Module):

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError(
                "FSDP requires initialized process group"
            )

        self.module = module

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.compute_dtype = compute_dtype

        self.sharded_params: dict[int, ShardedParameter] = {}

        # 等待中的异步通信
        self.pending_sharded_grads = []
        self.pending_replicated_grads = []

        # ---------------------------
        # 1. 初始模型保持一致
        # ---------------------------
        self._broadcast_parameters()

        # ---------------------------
        # 2. 参数分片
        # ---------------------------
        self._shard_parameters()

        # ---------------------------
        # 3. 安装 hooks
        # ---------------------------
        self._register_hooks()

    # =========================================================
    # 初始化
    # =========================================================

    @torch.no_grad()
    def _broadcast_parameters(self):

        for param in self.module.parameters():

            dist.broadcast(
                param.data,
                src=0,
            )

    def _is_shardable(self, module: nn.Module):

        from cs336_basics.model import Linear, Embedding

        return isinstance(
            module,
            (
                Linear,
                Embedding,
                nn.Linear,
                nn.Embedding,
            ),
        )

    @torch.no_grad()
    def _make_shard(
        self,
        param: nn.Parameter,
    ):

        original_shape = param.shape
        original_numel = param.numel()

        flat = param.data.reshape(-1)

        # 向上取整
        shard_numel = (
            original_numel
            + self.world_size
            - 1
        ) // self.world_size

        total_numel = (
            shard_numel
            * self.world_size
        )

        # padding
        padded = torch.zeros(
            total_numel,
            dtype=flat.dtype,
            device=flat.device,
        )

        padded[:original_numel].copy_(flat)

        start = (
            self.rank
            * shard_numel
        )

        end = (
            start
            + shard_numel
        )

        local_shard = (
            padded[start:end]
            .clone()
        )

        # -----------------------------
        # 关键：
        #
        # param.data 不再是完整参数
        #
        # 而是本 rank 的 shard
        # -----------------------------
        param.data = local_shard

        info = ShardedParameter(
            param=param,
            original_shape=original_shape,
            original_numel=original_numel,
            shard_numel=shard_numel,
            local_shard=local_shard,
        )

        self.sharded_params[id(param)] = info

    @torch.no_grad()
    def _shard_parameters(self):

        for module in self.module.modules():

            if not self._is_shardable(module):
                continue

            if not hasattr(module, "weight"):
                continue

            param = module.weight

            self._make_shard(param)

    # =========================================================
    # AllGather
    # =========================================================

    @torch.no_grad()
    def _all_gather_parameter(
        self,
        info: ShardedParameter,
    ):

        local = info.local_shard

        # mixed precision:
        # master shard 还是 FP32
        #
        # 通信 / compute 使用低精度
        if self.compute_dtype is not None:

            communication_shard = (
                local.to(
                    self.compute_dtype
                )
            )

        else:

            communication_shard = local

        gathered = [
            torch.empty_like(
                communication_shard
            )
            for _ in range(
                self.world_size
            )
        ]

        # ===================================
        # shard
        #
        #       ↓ AllGather
        #
        # full parameter
        # ===================================

        dist.all_gather(
            gathered,
            communication_shard,
        )

        full = torch.cat(
            gathered,
            dim=0,
        )

        # 去掉 padding
        full = full[
            :info.original_numel
        ]

        full = full.view(
            info.original_shape
        )

        # 临时把 Parameter 替换为完整版本
        info.param.data = full

    @torch.no_grad()
    def _restore_parameter_shard(
        self,
        info: ShardedParameter,
    ):

        # =================================
        # full parameter
        #
        #       ↓ 丢掉
        #
        # local shard
        # =================================

        info.param.data = (
            info.local_shard
        )

    # =========================================================
    # Gradient ReduceScatter
    # =========================================================

    @torch.no_grad()
    def _reduce_scatter_gradient(
        self,
        param: nn.Parameter,
    ):

        if param.grad is None:
            return

        info = self.sharded_params[
            id(param)
        ]

        full_grad = (
            param.grad
            .detach()
            .reshape(-1)
        )

        # padding
        total_numel = (
            info.shard_numel
            * self.world_size
        )

        padded_grad = torch.zeros(
            total_numel,
            dtype=full_grad.dtype,
            device=full_grad.device,
        )

        padded_grad[
            :info.original_numel
        ].copy_(full_grad)

        # 完整 gradient 已经不需要了
        param.grad = None

        # parameter 恢复成本地 shard
        self._restore_parameter_shard(
            info
        )

        backend = str(
            dist.get_backend()
        )

        # ==========================================
        # NCCL 等：
        # 真正 ReduceScatter
        # ==========================================

        if (
            self.world_size > 1
            and "gloo" not in backend.lower()
        ):

            output = torch.empty(
                info.shard_numel,
                dtype=padded_grad.dtype,
                device=padded_grad.device,
            )

            work = (
                dist.reduce_scatter_tensor(
                    output,
                    padded_grad,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
            )

            self.pending_sharded_grads.append(
                (
                    work,
                    info,
                    output,
                    None,
                )
            )

        else:

            # ==================================
            # Gloo fallback
            #
            # AllReduce(full gradient)
            # +
            # 自己切对应 shard
            #
            # 数学上等价 ReduceScatter
            # ==================================

            work = dist.all_reduce(
                padded_grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )

            self.pending_sharded_grads.append(
                (
                    work,
                    info,
                    None,
                    padded_grad,
                )
            )

    # =========================================================
    # 没有 shard 的小参数
    #
    # 例如 RMSNorm
    # =========================================================

    @torch.no_grad()
    def _all_reduce_replicated_gradient(
        self,
        param: nn.Parameter,
    ):

        if param.grad is None:
            return

        work = dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        self.pending_replicated_grads.append(
            (
                work,
                param,
            )
        )

    # =========================================================
    # Hooks
    # =========================================================

    def _register_hooks(self):

        sharded_ids = set(
            self.sharded_params.keys()
        )

        # ----------------------------------------
        # Linear / Embedding
        # ----------------------------------------

        for module in self.module.modules():

            if not self._is_shardable(module):
                continue

            if not hasattr(module, "weight"):
                continue

            param = module.weight

            info = self.sharded_params[
                id(param)
            ]

            # =========================
            # forward 前：
            #
            # AllGather parameter
            # =========================

            def forward_pre_hook(
                mod,
                inputs,
                info=info,
            ):

                self._all_gather_parameter(
                    info
                )

            # =========================
            # forward 后：
            #
            # 释放 full parameter
            # =========================

            def forward_post_hook(
                mod,
                inputs,
                output,
                info=info,
            ):

                self._restore_parameter_shard(
                    info
                )

            # =========================
            # backward 前：
            #
            # 再次 AllGather parameter
            # =========================

            def backward_pre_hook(
                mod,
                grad_output,
                info=info,
            ):

                self._all_gather_parameter(
                    info
                )

            module.register_forward_pre_hook(
                forward_pre_hook
            )

            module.register_forward_hook(
                forward_post_hook
            )

            module.register_full_backward_pre_hook(
                backward_pre_hook
            )

            # =========================
            # 完整 gradient 出现以后：
            #
            # ReduceScatter
            # =========================

            param.register_post_accumulate_grad_hook(
                self._reduce_scatter_gradient
            )

        # ----------------------------------------
        # RMSNorm 等 replicated 参数
        # ----------------------------------------

        for param in self.module.parameters():

            if id(param) in sharded_ids:
                continue

            if not param.requires_grad:
                continue

            param.register_post_accumulate_grad_hook(
                self._all_reduce_replicated_gradient
            )

    # =========================================================
    # optimizer.step 前调用
    # =========================================================

    @torch.no_grad()
    def finish_gradient_synchronization(
        self,
    ):

        # ====================================
        # 等待 ReduceScatter
        # ====================================

        for (
            work,
            info,
            output,
            fallback_buffer,
        ) in self.pending_sharded_grads:

            work.wait()

            if output is not None:

                local_grad = output

            else:

                start = (
                    self.rank
                    * info.shard_numel
                )

                end = (
                    start
                    + info.shard_numel
                )

                local_grad = (
                    fallback_buffer[
                        start:end
                    ]
                    .clone()
                )

            # 求平均
            local_grad /= (
                self.world_size
            )

            # optimizer 要和 master weight
            # 使用相同 dtype
            local_grad = local_grad.to(
                info.local_shard.dtype
            )

            info.param.grad = (
                local_grad.view_as(
                    info.local_shard
                )
            )

        self.pending_sharded_grads.clear()

        # ====================================
        # replicated parameters AllReduce
        # ====================================

        for (
            work,
            param,
        ) in self.pending_replicated_grads:

            work.wait()

            param.grad /= (
                self.world_size
            )

        self.pending_replicated_grads.clear()

    # =========================================================
    # forward
    # =========================================================

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ):

        return self.module(
            *args,
            **kwargs,
        )

    # =========================================================
    # 重建完整参数
    # =========================================================

    @torch.no_grad()
    def gather_full_params(self):

        result = {}

        for (
            name,
            param,
        ) in self.module.named_parameters():

            if id(param) not in self.sharded_params:

                result[name] = (
                    param.detach().clone()
                )

                continue

            info = self.sharded_params[
                id(param)
            ]

            gathered = [
                torch.empty_like(
                    info.local_shard
                )
                for _ in range(
                    self.world_size
                )
            ]

            dist.all_gather(
                gathered,
                info.local_shard,
            )

            full = torch.cat(
                gathered,
                dim=0,
            )

            full = full[
                :info.original_numel
            ]

            result[name] = (
                full
                .view(info.original_shape)
                .clone()
            )

        return result