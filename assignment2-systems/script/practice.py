import math
import torch
import triton.language as tl
import triton

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
    query_tile_index=tl.program_id(0)
    batch_index=tl.program_id(1)

    Q_block_ptr=tl.make_block_ptr(
        Q_ptr+batch_index*stride_qb,
        shape=(N_QUERIES,D),
        strides=(stride_qq,stride_qd),
        offsets=(
            query_tile_index*Q_TILE_SIZE,
            0
        ),
        block_shape=(Q_TILE_SIZE,D),
        order=(1,0),
    )
    K_block_ptr=tl.make_block_ptr(
        K_ptr+batch_index*stride_kb,
        shape=(N_KEYS,D),
        strides=(stride_kk,stride_kd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE,D),
        order=(1,0),
    )
    V_block_ptr=tl.make_block_ptr(
        V_ptr+batch_index*stride_vb,
        shape=(N_KEYS,D),
        strides=(stride_vk,stride_vd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE,D),
        order=(1,0),
    )
    O_block_ptr=tl.make_block_ptr(
        O_ptr+batch_index*stride_ob,
        shape=(N_QUERIES,D),
        strides=(stride_oq,stride_od),
        offsets=(
            query_tile_index*Q_TILE_SIZE,
            0
        ),
        block_shape=(Q_TILE_SIZE,D),
    )
    L_block_ptr=tl.make_block_ptr(
        L_ptr+batch_index*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q=tl.load(Q_block_ptr)


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

    for k_start in range(0,N_KEYS,K_TILE_SIZE):
        K=tl.load(K_block_ptr)
        V=tl.load(V_block_ptr)

        S=tl.dot(
            Q,
            tl.trans(K),
        )*scale

        if IS_CAUSAL:
            k_indices=(
                k_start*K_TILE_SIZE+
                tl.arange(0,K_TILE_SIZE)
            )

            causal_mask=(
                q_indices[:,None]>
                k_indices[None,:]
            )

            S=tl.where(
                causal_mask,
                S,
                -1.0e6,
            )

        # online softmax
        tile_max=tl.max(S,axis=1)
        m_max=tl.maximum(m,tile_max)

        alpha=tl.exp(m-m_max)

        P_tilde=tl.exp(S-m_max[:,None])

        l=(l*alpha+tl.sum(P_tilde,axis=1))


        O_acc=(O_acc*alpha[:,None])

        P_for_dot = P_tilde.to(V.dtype)

        O_acc=tl.dot(
            P_for_dot,
            V,
            acc=O_acc,
        )



        m=m_max
        K_block_ptr=K_block_ptr.advance(K_TILE_SIZE,0)
        V_block_ptr=V_block_ptr.advance(K_TILE_SIZE,0)

    O_acc=(
        O_acc/
        l[:,None]
    )

    L=m+tl.log(l)

    tl.store(
        O_block_ptr,
        O_acc.to(O_block_ptr.type.element_ty)
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

        ctx.save_for_backward(
            Q,
            K,
            V,
            O,
            L,
        )

        ctx.is_causal=bool(is_causal)

        return O
    
    def backward(ctx):
        raise NotImplementedError(
            "FlashAttention backward is implemented in the next problem"
        )