import torch
import torch.distributed as dist
import torch.nn as nn

class DDPOverlapIndividualParameters(nn.Module):
    def __init__(self,module:nn.Module):
        super().__init__()

        self.module=module
        self.world_size=dist.get_world_size()

        self.handles=[]

        for param in module.parameters():
            dist.broadcast(param.data,src=0)

        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_hook()
                )


    def _make_hook(self):
        def hook(param:torch.Tensor):
            # 此时param计算完成
            handle=dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            self.handles.append((handle,param))
        return hook
    

    def forward(self,*args,**kwargs):
        return self.module(*args,**kwargs)
    
    def finish_gradient_synchronization(self):
        for handle,param in self.handles:
            handle.wait()
            param.grad.div_(self.world_size)
        self.handles.clear()