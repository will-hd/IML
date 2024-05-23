import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product(q, k, v):
    """
    Softmax[qk^T/sqrt{d_k}] v
    Parameters
    ----------
    q : torch.Tensor
        Shape (batch_size, num_heads,  SeqLen, d_k) 
    k : torch.Tensor
        Shape (batch_size, num_heads,  SeqLen, d_k) 
    v : torch.Tensor
        Shape (batch_size, num_heads,  SeqLen, d_k) 
    Returns
    -------
    """
    d_k = q.size()[-1]

    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)

    return values, attention

class MultiheadAttention(nn.Module):
    """
    d_k = d_v = x_dim / num_heads
    """

    def __init__(self, input_dim: int, embed_dim: int, num_heads: int):
        """

        embed_dim : int
            d_k if were to use 
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, input_dim)
        Returns
        -------
        o : torch.Tensor
            Shape (batch_size, num_points, embed_dim)
            Often set embed_dim = input_dim
        """
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim) # 'Concatenates' heads along Dims dimension
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class CrossAttention(nn.Module):
    
    def __init__(self,
                 query_dim: int,
                 key_dim: int,
                 value_dim: int,
                 embed_dim: int, # = out_dim
                 num_heads: int
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(value_dim, embed_dim)

        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def expand_heads(self, t, batch_size, seq_length):
        t = t.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        t = t.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        return t

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                return_attention: bool = False):
        """
        Note that querys = targets
        Parameters
        ----------
        query : torch.Tensor
            Shape (batch_size, query_length, query_dim)
        key : torch.Tensor
            Shape (batch_size, key_length, key_dim)
        value : torch.Tensor
            Shape (batch_size, value_length, value_dim)
        """
        batch_size, query_length, _ = query.size()
        batch_sizek, key_length, _ = key.size()
        batch_sizev, value_length, _ = value.size()
        assert key_length == value_length
        assert batch_size == batch_sizek; assert batch_size == batch_sizev # Sanity

        q = self.q_proj(query).reshape(batch_size, query_length, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(batch_size, key_length, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(batch_size, value_length, self.num_heads, self.head_dim)

        q = self.expand_heads(q, batch_size, query_length) # Shape (batch_size, self.num_heads, query_length, self.head_dim)
        k = self.expand_heads(k, batch_size, key_length) 
        v = self.expand_heads(v, batch_size, value_length) 

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v) # Shape (batch_size, self.num_heads, self.query_length, self.head_dim)
        assert torch.Size((batch_size, self.num_heads, query_length, self.head_dim)) == values.size()
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, query_length, self.embed_dim) # 'Concatenates' heads along Dims dimension
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

