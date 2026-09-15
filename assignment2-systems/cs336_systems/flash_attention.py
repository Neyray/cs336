import math

import torch
import triton
import triton.language as tl


# ============================================================
# Part (a): Pure PyTorch FlashAttention-2 forward
# ============================================================

# 纯 PyTorch 参考实现，方便理解/对拍
class FlashAttentionPyTorch(torch.autograd.Function):
    """
    Pure-PyTorch tiled FlashAttention-2 forward.

    Used as a readable reference implementation for debugging
    the Triton kernel.
    """
    # 这个方法属于这个 class，但调用时不需要 self
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q: [B, Nq, D]
        K: [B, Nk, D]
        V: [B, Nk, D]

        Returns:
            O: [B, Nq, D]

        Also saves:
            Q, K, V, O, L

        where:
            L = logsumexp(S, dim=-1)
        """
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]

        scale = 1.0 / math.sqrt(d)

        # Assignment guarantees dimensions are powers of 2
        # and at least 16, so 16 is a safe tile size.
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        O = torch.empty_like(Q)

        # L should be kept in fp32 for numerical stability.
        L = torch.empty(
            (batch_size, n_queries),
            device=Q.device,
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # Each outer iteration corresponds to one query tile.
        # ----------------------------------------------------
        for q_start in range(0, n_queries, Q_TILE_SIZE):
            q_end = min(q_start + Q_TILE_SIZE, n_queries)

            q = Q[:, q_start:q_end, :]

            # Use fp32 accumulators.
            q_float = q.float()

            current_q_size = q_end - q_start

            # running max:
            # [B, Bq]
            m = torch.full(
                (batch_size, current_q_size),
                -float("inf"),
                device=Q.device,
                dtype=torch.float32,
            )

            # running softmax denominator:
            # [B, Bq]
            l = torch.zeros(
                (batch_size, current_q_size),
                device=Q.device,
                dtype=torch.float32,
            )

            # running unnormalized output:
            # [B, Bq, D]
            o_acc = torch.zeros(
                (batch_size, current_q_size, d),
                device=Q.device,
                dtype=torch.float32,
            )

            # ------------------------------------------------
            # Scan all K/V tiles.
            # ------------------------------------------------
            for k_start in range(0, n_keys, K_TILE_SIZE):
                k_end = min(k_start + K_TILE_SIZE, n_keys)

                k = K[:, k_start:k_end, :].float()
                v = V[:, k_start:k_end, :].float()

                # S_ij = Q_i K_j^T / sqrt(d)
                #
                # [B, Bq, D] @ [B, D, Bk]
                # ->
                # [B, Bq, Bk]
                S = torch.matmul(
                    q_float,
                    k.transpose(-1, -2),
                ) * scale

                # Optional causal masking.
                if is_causal:
                    q_indices = torch.arange(
                        q_start,
                        q_end,
                        device=Q.device,
                    )

                    k_indices = torch.arange(
                        k_start,
                        k_end,
                        device=Q.device,
                    )

                    causal_mask = (
                        q_indices[:, None]
                        >= k_indices[None, :]
                    )

                    S = torch.where(
                        causal_mask[None, :, :],
                        S,
                        torch.tensor(
                            -1e6,
                            device=S.device,
                            dtype=S.dtype,
                        ),
                    )

                # --------------------------------------------
                # Online softmax
                # --------------------------------------------

                # m_new = max(m_old, rowmax(S))
                tile_max = S.max(dim=-1).values
                m_new = torch.maximum(m, tile_max)

                # alpha = exp(m_old - m_new)
                alpha = torch.exp(m - m_new)

                # P_tilde = exp(S - m_new)
                P_tilde = torch.exp(
                    S - m_new[..., None]
                )

                # l_new =
                #   alpha * l_old
                #   + rowsum(P_tilde)
                l = (
                    alpha * l
                    + P_tilde.sum(dim=-1)
                )

                # O_new =
                #   alpha * O_old
                #   + P_tilde @ V
                o_acc = (
                    alpha[..., None] * o_acc
                    + torch.matmul(P_tilde, v)
                )

                m = m_new

            # ------------------------------------------------
            # Finish normalization.
            # ------------------------------------------------

            O[:, q_start:q_end, :] = (
                o_acc / l[..., None]
            ).to(Q.dtype)

            # L = m + log(l)
            L[:, q_start:q_end] = (
                m + torch.log(l)
            )

        # Required for the later backward pass.
        ctx.save_for_backward(
            Q,
            K,
            V,
            O,
            L,
        )

        ctx.is_causal = bool(is_causal)

        return O

    @staticmethod
    def backward(ctx, grad_output):
        # 4.2.2 only requires forward.
        # Backward belongs to the following problem.
        raise NotImplementedError(
            "FlashAttention backward is implemented in the next problem."
        )


# ============================================================
# Part (b/c): Triton FlashAttention-2 forward kernel
# ============================================================


# 真正的 Triton GPU kernel

# 这个 Python 写的函数不是普通 Python 函数，而是一个 Triton GPU kernel，要由 Triton 编译成 GPU 可以执行的代码
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
# 计算内存地址
    stride_qb,     # batch 维移动一步，需要跳多少元素
    stride_qq,     # query 维移动一步
    stride_qd,     # embedding/head dimension 移动一步

    stride_kb,
    stride_kk,
    stride_kd,

    stride_vb,
    stride_vk,
    stride_vd,

    stride_ob,
    stride_oq,
    stride_od,

    stride_lb,
    stride_lq,

    N_QUERIES,     # Q sequence length
    N_KEYS,     # K sequence length
    scale,

    D: tl.constexpr,     # tl.constexpr表示这些值在 kernel 编译时就是已知常量
    Q_TILE_SIZE: tl.constexpr,     # 一个Q tile包含的行数，列数由D也就是weight的维度决定
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # ========================================================
    # One Triton program handles:
    #
    #   one batch
    #   +
    #   one Q tile
    # ========================================================

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # --------------------------------------------------------
    # Q tile pointer
    # --------------------------------------------------------

    '''
    Q[batch_index,
    q_start:q_start + Q_TILE_SIZE,
    0:D]
    '''
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        # 向下一行 query 走一步 → 地址加 stride_qq
        # 向右一个 d 走一步 → 地址加 stride_qd
        strides=(stride_qq, stride_qd),

        # 这个 tile 从整个矩阵哪里开始，列为零是因为Q0要逐一与k1,k2...相乘
        offsets=(
            query_tile_index * Q_TILE_SIZE,
            0,
        ),

        # 要拿多大的块
        block_shape=(Q_TILE_SIZE, D),
        # 告诉Trition这个 block 内部各维度的内存访问/布局顺序
        order=(1, 0),
    )

    # --------------------------------------------------------
    # K starts from the first key tile.
    # --------------------------------------------------------

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # --------------------------------------------------------
    # V starts from the first key tile.
    # --------------------------------------------------------

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # --------------------------------------------------------
    # Output tile corresponding to this Q tile.
    # --------------------------------------------------------

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(
            query_tile_index * Q_TILE_SIZE,
            0,
        ),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # --------------------------------------------------------
    # L contains one scalar per query.
    # --------------------------------------------------------

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(
            query_tile_index * Q_TILE_SIZE,
        ),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # ========================================================
    # Load Q once.
    #
    # This Q tile remains on-chip while we scan K/V tiles.
    # ========================================================
    '''
HBM / global memory
↓
GPU program 的片上工作数据
    '''
    Q = tl.load(Q_block_ptr)

    # ========================================================
    # Online-softmax state.
    # Keep these in fp32.
    # ========================================================

    # m: [Bq]
    m = tl.full(
        (Q_TILE_SIZE,),
        -float("inf"),
        tl.float32,
    )

    # l: [Bq]
    l = tl.zeros(
        (Q_TILE_SIZE,),
        dtype=tl.float32,
    )

    # O accumulator: [Bq, D]
    O_acc = tl.zeros(
        (Q_TILE_SIZE, D),
        dtype=tl.float32,
    )

    # Query indices are constant for this program.
    q_indices = (
        query_tile_index * Q_TILE_SIZE
        + tl.arange(0, Q_TILE_SIZE)
    )

    # ========================================================
    # Single loop over K/V tiles.
    # ========================================================

    for k_start in range(
        0,
        N_KEYS,
        K_TILE_SIZE,
    ):
        # ----------------------------------------------------
        # Load one K/V tile from global memory.
        # ----------------------------------------------------

        K = tl.load(K_block_ptr)
        V = tl.load(V_block_ptr)

        # ----------------------------------------------------
        # S = Q K^T / sqrt(d)
        #
        # [Bq, D] @ [D, Bk]
        # ->
        # [Bq, Bk]
        # ----------------------------------------------------

        S = tl.dot(
            Q,
            tl.trans(K),
        ) * scale

        # ----------------------------------------------------
        # Part (c): causal mask
        # ----------------------------------------------------

        if IS_CAUSAL:
            k_indices = (
                k_start
                + tl.arange(0, K_TILE_SIZE)
            )

            causal_mask = (
                q_indices[:, None]
                >= k_indices[None, :]
            )

            S = tl.where(
                causal_mask,
                S,
                -1.0e6,
            )

        # ====================================================
        # Online softmax
        # ====================================================

        # rowmax of current tile
        tile_max = tl.max(
            S,
            axis=1,
        )

        # m_new
        m_new = tl.maximum(
            m,
            tile_max,
        )

        # alpha =
        # exp(m_old - m_new)
        alpha = tl.exp(
            m - m_new
        )

        # P_tilde =
        # exp(S - m_new)
        P_tilde = tl.exp(
            S - m_new[:, None]
        )

        # ----------------------------------------------------
        # Update denominator:
        #
        # l_new =
        # alpha * l_old
        # + rowsum(P_tilde)
        # ----------------------------------------------------

        l = (
            alpha * l
            + tl.sum(
                P_tilde,
                axis=1,
            )
        )

        # ----------------------------------------------------
        # Rescale previous output.
        # ----------------------------------------------------
        # 不是矩阵相乘，而是缩放
        O_acc = (
            O_acc
            * alpha[:, None]
        )

        # ----------------------------------------------------
        # Add contribution from current V tile.
        #
        # O += P_tilde @ V
        # ----------------------------------------------------

        P_for_dot = P_tilde.to(V.dtype)

        O_acc = tl.dot(
            P_for_dot,
            V,
            acc=O_acc,
        )

        # Advance running maximum.
        m = m_new

        # ----------------------------------------------------
        # Move K/V block pointers to next tile.
        # ----------------------------------------------------

        K_block_ptr = K_block_ptr.advance(
            (K_TILE_SIZE, 0)
        )

        V_block_ptr = V_block_ptr.advance(
            (K_TILE_SIZE, 0)
        )

    # ========================================================
    # Final normalization
    # ========================================================

    O_acc = (
        O_acc
        / l[:, None]
    )

    # logsumexp:
    #
    # L = m + log(l)
    L = m + tl.log(l)

    # ========================================================
    # Write results back to global memory.
    # ========================================================
    # 把 Triton kernel 算出来的数值写进 GPU 内存中的 Tensor
    tl.store(
        O_block_ptr,
        O_acc.to(
            O_block_ptr.type.element_ty
        ),
    )

    tl.store(
        L_block_ptr,
        L,
    )


# ============================================================
# Autograd wrapper around Triton kernel
# ============================================================

# 从 Python 启动 kernel 的包装器
# 负责准备输入和启动 kernel
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        is_causal=False,
    ):
        if not Q.is_cuda:
            raise RuntimeError(
                "FlashAttentionTriton requires CUDA tensors."
            )

        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]

        O = torch.empty_like(Q)

        L = torch.empty(
            (batch_size, n_queries),
            device=Q.device,
            dtype=torch.float32,
        )

        # 16 is safe because the assignment guarantees
        # dimensions are powers of 2 and >= 16.
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        # Number of query tiles.
        # 向上取整
        n_query_tiles = triton.cdiv(
            n_queries,
            Q_TILE_SIZE,
        )

        # ----------------------------------------------------
        # grid:
        #
        # dimension 0 -> query tile
        # dimension 1 -> batch
        #
        # one program =
        # one batch + one query tile
        # ----------------------------------------------------

        # 表示program个数，第一维是Q tiles，第二维是batch size
        grid = (
            n_query_tiles,
            batch_size,
        )

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,

            Q.stride(0),
            Q.stride(1),
            Q.stride(2),

            K.stride(0),
            K.stride(1),
            K.stride(2),

            V.stride(0),
            V.stride(1),
            V.stride(2),

            O.stride(0),
            O.stride(1),
            O.stride(2),

            L.stride(0),
            L.stride(1),

            n_queries,
            n_keys,

            1.0 / math.sqrt(d),

            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=is_causal,
        )

        # Save exactly one [B, Nq] tensor: L.
        # The existing tests inspect saved_tensors.
        # 告诉 PyTorch autograd：这些 Tensor 在 backward 时还要用，请保留下来，并让我届时能取到
        ctx.save_for_backward(
            Q,
            K,
            V,
            O,
            L,
        )

        ctx.is_causal = bool(is_causal)

        return O

    @staticmethod
    def backward(ctx, grad_output):
        # This is the next problem, not 4.2.2.
        raise NotImplementedError(
            "FlashAttention backward is implemented in the next problem."
        )