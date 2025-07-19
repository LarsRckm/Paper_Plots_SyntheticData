import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        result = self.embedding(x) * math.sqrt(self.d_model)
        return result

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #create matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #create vector of shape (seq_len,1)
        position = torch.arange(0,seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model, 2).float()* 2/d_model * (-math.log(10000.0))) #(1, d_model/2)
        #apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        pe = pe.unsqueeze(dim=0) #(1,seq_len, d_model)

        # delattr(self, 'pe')
        self.register_buffer('pe', pe, persistent=True)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad_(False) #add the positional encodings based on the the different sequence lengths (x.shape[1])
        x = self.dropout(x)
        return x

class LayerNormalisation(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim= -1, keepdim = True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, dtype=torch.float32)
        # self.l_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        #(Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h:int , dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, 'd_model is not divisible by h'
        
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False) # WQ
        self.w_k = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False) # WK
        self.w_v = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False) # WV
        self.w_o = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False) # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # save_csv(mask, "mask")
        # save_csv(query, "query")
        # save_csv(key, "key")
        # save_csv(value, "value")
        

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)#(1,8,2,2)
        # save_csv(attention_scores, "attention_scores_before")
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -float("inf"))
            # save_csv(attention_scores, "attention_scores_mask_1")
        attention_scores = attention_scores.softmax(dim= -1) #(batch, h, seq_len, seq_len)
        # save_csv(attention_scores, "attention_scores_softmax")
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # result = (attention_scores @ value)
        # result = torch.nan_to_num(result)
        return (attention_scores @ value), attention_scores #(batch, h, seq_len, d_k)


    def forward(self, q,k,v,mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)-->(1,1,512)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model) -->(1,999,512)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model) -->(1,999,512)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k) -->(8, 8, seq_len, 64)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) #(1,8,2,64)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)#-->(1,8,2,64)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)#(1,8,2,64)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # save_csv(x, "x")

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class MultiHeadAttentionBlock_new(nn.Module):
    def __init__(self, d_model: int, h:int , dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, 'd_model is not divisible by h'
        
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, dtype=torch.float32) # WQ
        self.w_k = nn.Linear(d_model, d_model, dtype=torch.float32) # WK
        self.w_v = nn.Linear(d_model, d_model, dtype=torch.float32) # WV
        self.w_o = nn.Linear(d_model, d_model, dtype=torch.float32) # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)#(1,8,2,2)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e20)
            # save_csv(attention_scores, "attention_scores")
        attention_scores = attention_scores.softmax(dim= -1) #(batch, h, seq_len, seq_len)
        # save_csv(attention_scores, "attention_scores_softmax")
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores #(batch, h, seq_len, d_k)


    def forward(self, q,k,v,mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)-->(1,1,512)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model) -->(1,999,512)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model) -->(1,999,512)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k) -->(8, 8, seq_len, 64)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) #(1,8,2,64)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)#-->(1,8,2,64)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)#(1,8,2,64)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask)) # wieso bei einem mit lambda und bei dem anderen ohne lambda
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()
    
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList ([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self,x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionsLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self,x):
        #(batch,seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

class Transformer_Encoder_Interpolation_CE(nn.Module):
    def __init__(self, encoder: Encoder, src_emb: InputEmbedding, src_pos: PositionalEncoding, projection_layer: ProjectionsLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_emb = src_emb
        self.src_pos = src_pos
        self.projections_layer = projection_layer


    def encode(self, src, src_mask):
        src = self.src_emb(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def project(self,x):
        return self.projections_layer(x)

#model for interpolation with encoder and index input values; projection Layer projects output to index probability distribution
def build_encoder_interpolation_uknToken_projection(vocab_size:int,src_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2024)-> Transformer_Encoder_Interpolation_CE:
    src_embed = InputEmbedding(d_model, vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)

    encoder_blocks = []

    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    projection_layer = ProjectionsLayer(d_model, vocab_size)

    transformer = Transformer_Encoder_Interpolation_CE(encoder, src_embed, src_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer