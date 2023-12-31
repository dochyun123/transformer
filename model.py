import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_model = 256 # embedding dimension
h = 8 # head size
max_len = 100 # max sequence length



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
    def __init__(self, d_embed, max_len, device=device):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed)
        )
        encoding[:, 0::2] = torch.sin(pos * div_term)
        encoding[:, 1::2] = torch.cos(pos * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        seq_len = x.size(-2)
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out


def dot_product_attention(Q, K, V, mask=None):
    d_k = K.size(-1)  # (batch_size,key length, key demension)
    softmax = torch.nn.Softmax(dim=-1)
    scores = torch.matmul(Q, K.T) / (math.sqrt(d_k))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-1e9"))

    attention_prob = softmax(scores)
    attention = torch.matmul(attention_prob, V)
    return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # embedding size : 512
        self.h = h  # head size : 8
        assert d_model % h == 0
        self.head_dim = d_model // h  # head_dim = 64
        self.v_pr = nn.Linear(
            self.d_model, self.d_model, bias=False
        )  # fully connect linear
        self.k_pr = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_pr = nn.Linear(self.d_model, self.d_model, bias=False)
        self.fc_out = nn.Linear(self.d_model, self.d_model)

    def forward(self, values, keys, query, mask):
        batch_size = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # query = [batch size, query len, head dim]
        # key = [batch size, key len, head dim]
        # value = [batch size, value len, head dim]
        Q = self.q_pr(query)
        K = self.k_pr(keys)
        V = self.v_pr(values)

        values = Q.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = K.reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = V.reshape(batch_size, query_len, self.heads, self.head_dim)

        attention = dot_product_attention(queries, keys, values, mask)

        # Concatenate
        attention = attention.reshape(batch_size, query_len, self.heads * self.head_dim)
        out = self.fc_out(attention)
        return out
    

class AddNorm(nn.Module): 
    
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderLayer(nn.Module):
    def __init__(self, head_dim, n_heads, filter, dropout, device):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(head_dim)
        self.ff_layer_norm = nn.LayerNorm(head_dim)
        self.multihead_attention = MultiHeadAttention(head_dim, n_heads, dropout, device)
        self.positionwise_ffn = FFN(head_dim,filter,dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, mask):             
        
        out = self.multihead_attention(X, X, X, mask)
        X = self.layer_norm(X + self.dropout(out))
        out = self.positionwise_ffn(X)
        X = self.ff_layer_norm(X + self.dropout(out))
    
        return X


class Encoder(nn.Module):
    def __init__(self, N ,d_embed, max_length, vocab_size, head_dim, n_heads, filter, dropout, device):
        super().__init__()

        self.device = device
        self.tok_embedding = TokenEmbedding(d_embed, vocab_size)
        self.pos_embedding = PositionalEncoding(d_embed,max_length)

        self.layers = nn.ModuleList([EncoderLayer(head_dim, n_heads, filter, dropout, device) 
                                     for _ in range(N)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X, mask):
            
            batch_size = X.shape[0]
            X_len = X.shape[1]
            
            pos = torch.arange(0, X_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            
            #pos = [batch size, src len]
            
            X = self.dropout(self.tok_embedding(X)  + self.pos_embedding(pos))
        
            for layer in self.layers:
                X = layer(X, mask)
                
            return X   
