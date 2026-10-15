import torch
import torch.nn as nn
from MultiAttentionHead import MultiAttentionHead
from utils import PositionwiseFeedForward,SublayerConnection
from Embeddings import TargetEmbedding,SourceEmbedding
from mask import create_padding_mask,create_decoder_mask
class EncoderLayer(nn.Module):
    def __init__(self, d_model,h,d_ff,dropout):
        
        super(EncoderLayer,self).__init__()
        self.self_attn=MultiAttentionHead(d_model,h)
        self.feed_forword=PositionwiseFeedForward(d_model,d_ff)
        self.sublayers=nn.ModuleList([SublayerConnection(d_model,dropout) for _ in range(2)])
        self.d_model=d_model
    def forward(self,x,src_mask):

        x=self.sublayers[0](x,lambda x: self.self_attn(x,x,x,src_mask))
        x=self.sublayers[1](x,self.feed_forword)
        return x


class DecoderLayer(nn.Module):
    def __init__(self,d_model,h,d_ff,dropout):
        super(DecoderLayer,self).__init__()
        self.self_attn=MultiAttentionHead(d_model,h)
        self.cross_attn=MultiAttentionHead(d_model,h)
        self.feed_forward=PositionwiseFeedForward(d_model,d_ff,dropout)

        self.sublayers=nn.ModuleList([SublayerConnection(d_model,dropout) for _ in range(3)])
        self.d_model=d_model

    def forward(self,x,memory,src_mask,tgt_mask):

        x=self.sublayers[0](x,lambda x: self.self_attn(x,x,x,tgt_mask))

        x=self.sublayers[1](x,lambda x: self.cross_attn(x,memory,memory,src_mask))

        x=self.sublayers[2](x,self.feed_forward)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model,N,h,d_ff,dropout=0.1):
        super(Encoder,self).__init__()
        self.layers=nn.ModuleList([EncoderLayer(d_model,h,d_ff,dropout) for _ in range(N)])

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return x



class Decoder(nn.Module):
    def __init__(self,d_model,N,h,d_ff,dropout=0.1):
        super(Decoder,self).__init__()
        self.layers=nn.ModuleList([
            DecoderLayer(d_model,h,d_ff,dropout) for _ in range(N)
        ])

    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return x




class Transformer(nn.Module):
    def __init__(self, src_vocab_size,tgt_vocab_size,d_model,N,h,d_ff,dropout=0.1):
        
        super(Transformer,self).__init__()

        self.src_embedding=SourceEmbedding(src_vocab_size,d_model)
        self.tgt_embedding=TargetEmbedding(tgt_vocab_size,d_model)

        self.encoder=Encoder(d_model,N,h,d_ff,dropout)
        self.decoder=Decoder(d_model,N,h,d_ff,dropout)

        self.fc_out=nn.Linear(d_model,tgt_vocab_size)

    def forward(self,src,tgt):

        src_mask=create_padding_mask(src)
        tgt_mask=create_decoder_mask(tgt)

        enc_output=self.encoder(self.src_embedding(src),src_mask)
        dec_output=self.decoder(self.tgt_embedding(tgt),enc_output,src_mask,tgt_mask)

        output=self.fc_out(dec_output)
        return output


    
        
