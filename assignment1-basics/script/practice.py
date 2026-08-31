import numpy as np
import numpy.typing as npt
import torch

def get_batch(
    dataset:npt.NDArray,
    batch_size:int,
    context_length:int,
    device=None,
)->tuple[torch.Tensor,torch.Tensor]:
    
    start_indices=torch.randint(
        low=0,
        high=len(dataset)-context_length,
        size=batch_size,
    )

    x=torch.stack([
        torch.from_numpy(
            dataset[i:i+context_length].astype(np.int64)
            for i in start_indices
        )
    ])

    y=torch.stack([
        torch.from_numpy(
            dataset[i+1:i+context_length+1].astype(np.int64)
            for i in start_indices
        )
    ])

    inputs=x.to(device)
    targets=y.to(device)

    return inputs,targets 