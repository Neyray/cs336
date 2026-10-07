"""Regression coverage beyond the original assignment attention tests."""

import pytest
import torch

from .adapters import (
    get_flashattention_autograd_function_pytorch,
    get_flashattention_autograd_function_triton,
)
from .test_attention import _attention_and_lse


_IMPLEMENTATIONS = [
    pytest.param(get_flashattention_autograd_function_pytorch, "cpu", id="pytorch"),
    pytest.param(
        get_flashattention_autograd_function_triton,
        "cuda",
        id="triton",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires a CUDA GPU"),
    ),
]


@pytest.mark.parametrize("factory,device", _IMPLEMENTATIONS)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("n_queries,n_keys", [(16, 32), (32, 16)])
def test_flash_rectangular_strided_inputs(factory, device, is_causal, n_queries, n_keys):
    """Exercise both causal directions and noncontiguous tensor strides."""
    torch.manual_seed(17)
    batch_size, head_dim = 2, 16
    q = torch.randn(batch_size, head_dim, n_queries, device=device).transpose(1, 2).requires_grad_()
    k = torch.randn(batch_size, head_dim, n_keys, device=device).transpose(1, 2).requires_grad_()
    v = torch.randn(batch_size, head_dim, n_keys, device=device).transpose(1, 2).requires_grad_()
    do = torch.randn(batch_size, head_dim, n_queries, device=device).transpose(1, 2)
    reference_inputs = [t.detach().clone().requires_grad_() for t in (q, k, v)]

    output = factory().apply(q, k, v, is_causal)
    expected, expected_lse = _attention_and_lse(*reference_inputs, is_causal)
    saved_lse = [t for t in output.grad_fn.saved_tensors if t.shape == (batch_size, n_queries)]
    assert len(saved_lse) == 1
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(saved_lse[0], expected_lse, rtol=1e-2, atol=1e-2)

    output.backward(do)
    expected.backward(do)
    for actual, reference in zip((q, k, v), reference_inputs):
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad, reference.grad, rtol=1e-2, atol=1e-2)
    if is_causal and n_keys > n_queries:
        # Keys beyond the last query cannot affect any output.
        assert torch.count_nonzero(k.grad[:, n_queries:]) == 0
        assert torch.count_nonzero(v.grad[:, n_queries:]) == 0


@pytest.mark.parametrize("factory,device", _IMPLEMENTATIONS)
def test_flash_default_causal_flag_with_only_query_grad(factory, device):
    """The optional fourth input may be omitted, and K/V need not need gradients."""
    torch.manual_seed(23)
    q = torch.randn(2, 16, 16, device=device, requires_grad=True)
    k = torch.randn(2, 32, 16, device=device)
    v = torch.randn(2, 32, 16, device=device)
    reference_q = q.detach().clone().requires_grad_()
    do = torch.randn_like(q)
    output = factory().apply(q, k, v)
    expected, _ = _attention_and_lse(reference_q, k, v)
    output.backward(do)
    expected.backward(do)
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(q.grad, reference_q.grad, rtol=1e-2, atol=1e-2)
    assert k.grad is None
    assert v.grad is None
