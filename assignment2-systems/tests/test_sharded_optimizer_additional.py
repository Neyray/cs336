"""Exercise parameter-group and closure contracts of optimizer sharding."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .adapters import get_sharded_optimizer
from .common import _cleanup_process_group, _setup_process_group


@pytest.mark.parametrize("optimizer_cls", [torch.optim.AdamW, torch.optim.SGD])
@pytest.mark.parametrize("initial_parameter_count", [1, 2])
def test_sharded_optimizer_param_groups_and_closure(optimizer_cls, initial_parameter_count):
    mp.spawn(_check_param_groups, args=(2, optimizer_cls, initial_parameter_count), nprocs=2, join=True)


def _check_param_groups(rank, world_size, optimizer_cls, initial_parameter_count):
    device = _setup_process_group(rank, world_size, backend="gloo")
    try:
        torch.manual_seed(31)
        sharded_params = [torch.nn.Parameter(torch.randn(3, device=device)) for _ in range(3)]
        reference_params = [torch.nn.Parameter(p.detach().clone()) for p in sharded_params]
        kwargs = {"lr": 0.1, "weight_decay": 0.01}
        if optimizer_cls is torch.optim.SGD:
            kwargs["momentum"] = 0.9

        def groups(params):
            return [
                {"params": [p], "lr": 0.02 * (i + 1), "weight_decay": 0.03 * i}
                for i, p in enumerate(params[:initial_parameter_count])
            ]

        sharded = get_sharded_optimizer(groups(sharded_params), optimizer_cls, **kwargs)
        reference = optimizer_cls(groups(reference_params), **kwargs)
        active_count = initial_parameter_count

        def closure(optimizer, params):
            def evaluate():
                optimizer.zero_grad(set_to_none=True)
                loss = sum(p.square().sum() for p in params[:active_count])
                loss.backward()
                return loss

            return evaluate

        for iteration in range(4):
            if iteration == 1:
                # The updated public learning rate must reach the local optimizer.
                sharded.param_groups[0]["lr"] = 0.007
                reference.param_groups[0]["lr"] = 0.007
            if iteration == 2:
                new_group = {"lr": 0.013, "weight_decay": 0.11}
                sharded.add_param_group({"params": [sharded_params[active_count]], **new_group})
                reference.add_param_group({"params": [reference_params[active_count]], **new_group})
                active_count += 1

            actual_loss = sharded.step(closure(sharded, sharded_params))
            expected_loss = reference.step(closure(reference, reference_params))
            torch.testing.assert_close(actual_loss, expected_loss)
            for actual, expected in zip(sharded_params, reference_params):
                torch.testing.assert_close(actual, expected)
                all_ranks = [torch.empty_like(actual) for _ in range(world_size)]
                dist.all_gather(all_ranks, actual.detach())
                for peer in all_ranks:
                    torch.testing.assert_close(actual, peer)

            # Momentum/AdamW states must exist only for this rank's parameters.
            local_state = sharded.local_optimizer.state
            active_params = sharded_params[:active_count]
            assert set(local_state).issubset(set(active_params))
            local_indices = [i for i, p in enumerate(active_params) if p in local_state]
            state_indices = [None] * world_size
            dist.all_gather_object(state_indices, local_indices)
            # Every active parameter has state on exactly one rank.
            assert sorted(i for indices in state_indices for i in indices) == list(range(active_count))
            sharded.zero_grad(set_to_none=False)
            for p in sharded_params[:active_count]:
                assert p.grad is not None
                assert torch.count_nonzero(p.grad) == 0
    finally:
        _cleanup_process_group()
