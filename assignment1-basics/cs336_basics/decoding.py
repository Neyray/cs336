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

    ids = tokenizer.encode(prompt)

    ids = torch.tensor(ids).unsqueeze(0)


    for _ in range(max_tokens):

        # forward
        logits = model(ids)


        # 最后一个位置
        logits = logits[:, -1, :]


        # temperature
        logits = logits / temperature


        probs = torch.softmax(logits, dim=-1)


        # top-p
        if top_p < 1.0:

            sorted_probs, sorted_ids = torch.sort(
                probs,
                descending=True
            )

            cumulative = torch.cumsum(
                sorted_probs,
                dim=-1
            )


            mask = cumulative > top_p

            mask[...,1:] = mask[...,:-1].clone()

            mask[...,0] = False


            sorted_probs[mask] = 0


            sorted_probs /= sorted_probs.sum()


            next_token = sorted_ids[
                torch.multinomial(sorted_probs,1)
            ]


        else:

            next_token = torch.multinomial(
                probs,
                1
            )


        ids = torch.cat(
            [ids,next_token],
            dim=1
        )


        if next_token.item()==tokenizer.eos_id:
            break


    return tokenizer.decode(
        ids[0].tolist()
    )