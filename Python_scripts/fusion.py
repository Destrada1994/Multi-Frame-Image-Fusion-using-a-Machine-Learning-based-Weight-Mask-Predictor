import torch

def Linear_Fusion(images,weights,cdim):
    weights_ex = weights.expand((-1,cdim,-1,-1,-1))
        
    out = images*weights_ex
    out = torch.sum(out,dim=2)
    return out







    
