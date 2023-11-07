# atttention

import torch
import torch.nn as nn
import math


def dot_product_attention(Q, K, V, mask):
    d = Q.size(-1)
    softmax = torch.nn.Softmax(dim=1)
    scores = torch.matmul(Q, K.T) / (math.sqrt(d))
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attention = softmax(dim=-1)(scores)
    return attention
