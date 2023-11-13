import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Position-wise Feed-Forward Networks
class FFN(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FFN, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# Embedding
class TokenEmbedding(nn.Module):

    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed


    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out

# Positional encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_embed, max_len=256, device= device):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(pos * div_term)
        encoding[:, 1::2] = torch.cos(pos * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)


    def forward(self, x):
        seq_len = x.size(-2)
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out



def dot_product_attention(Q, K, V, mask=None):
    d_k = K.size(-1)
    softmax = torch.nn.Softmax(dim=1)
    scores = torch.matmul(Q, K.T) / (math.sqrt(d_k))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    
    attention = softmax(dim=-1)(scores)
    return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        attention = dot_product_attention(queries, keys, values, mask)

        # Concatenate heads and put through final linear layer
        attention = attention.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(attention)
        return out