import torch.nn as nn
import torch
from einops import einsum

a = torch.tensor( [0,1,2,3] )

b = einsum(nn.functional.one_hot(a,10), [2 for i in range(10)], 'b n, n d -> b d')

print(b)

