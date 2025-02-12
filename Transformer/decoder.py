import torch
from torch import nn

from self_attention import SelfAttention
from encoder import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()

        self.attention_layer = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)

        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, keys, values, src_mask, trg_mask):

       
        attention = self.attention_layer.forward(x, x, x, trg_mask)
        queries = self.dropout(self.norm(attention + x))

       
        out = self.transformer_block(keys, values, queries, src_mask)
        
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size, 
                    heads, 
                    forward_expansion,
                    dropout,
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(out)

        return out

