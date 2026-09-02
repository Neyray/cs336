import torch


@torch.no_grad()
def decode(
    model,
    tokenizer,
    prompt,
    max_tokens=100,
    temperature=1.0,
    top_p=1.0,
):

    model.eval()

    # 模型所在设备（cpu / cuda），输入必须迁到同一设备
    device = next(model.parameters()).device

    # 上下文窗口上限（RoPE cache 只到 context_length）
    context_length = getattr(model, "context_length", None)

    ids = tokenizer.encode(prompt)

    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


    for _ in range(max_tokens):

        # 只喂最近 context_length 个 token，避免超出 RoPE 位置范围
        if context_length is not None and ids.shape[1] > context_length:
            model_input = ids[:, -context_length:]
        else:
            model_input = ids

        # forward
        logits = model(model_input)


        # 最后一个位置
        logits = logits[:, -1, :]


        # temperature
        logits = logits / temperature


        probs = torch.softmax(logits, dim=-1)


        # top-p (nucleus sampling)
        if top_p < 1.0:

            sorted_probs, sorted_ids = torch.sort(
                probs,
                descending=True
            )

            cumulative = torch.cumsum(
                sorted_probs,
                dim=-1
            )


            # 累积概率超过 top_p 的部分置零（保留刚好越过阈值的那个）
            mask = cumulative > top_p

            mask[..., 1:] = mask[..., :-1].clone()

            mask[..., 0] = False


            sorted_probs = sorted_probs.masked_fill(mask, 0.0)


            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)


            # 在排序后的分布上采样，再映射回原始 token id
            choice = torch.multinomial(sorted_probs, 1)

            next_token = torch.gather(sorted_ids, -1, choice)


        else:

            next_token = torch.multinomial(
                probs,
                1
            )


        ids = torch.cat(
            [ids, next_token],
            dim=1
        )


        if tokenizer.eos_id is not None and next_token.item() == tokenizer.eos_id:
            break


    return tokenizer.decode(
        ids[0].tolist()
    )
