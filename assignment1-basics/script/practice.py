import math

import torch

def get_lr_cosine_schedule(
    t:int,
    alpha_max:float,
    alpha_min:float,
    T_w:int,
    T_c:int,
)->float:
    if t<T_w:
        return t/T_w * alpha_max
    
    elif T_w<=t<=T_c:
        progress=(t-T_w)/(T_c-T_w)

        return alpha_min+0.5*(1+math.cos(math.pi*progress))*(alpha_max-alpha_min)
    
    else:
        return alpha_min