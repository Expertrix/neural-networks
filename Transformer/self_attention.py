import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by the number of heads"


        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)


        self.fc_out = nn.Linear(self.heads * self.head_dim, embed_size)

    def forward(self, keys, values, queries, mask=None):

        N = queries.shape[0]


        key_len, value_len, query_len = keys.shape[1], values.shape[1], queries.shape[1]

        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

 
        keys = self.keys(keys)
        values = self.values(values)
        queries = self.queries(queries)


        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))


        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", attention, values).reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out

