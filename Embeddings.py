# 嵌入层  输入的是序列  模型内部是张量  嵌入层就是起到转换作用
import torch
import torch.nn as nn
import math
from PositionalEncoding import PositionEncoding

class Embeddings(nn.Module):
    def __init__(self,vocab_size,d_model):
        super(Embeddings,self).__init__()
        self.embed=nn.Embedding(vocab_size,d_model)
        self.scale_factor=math.sqrt(d_model)

    def forward(self,x):
        return self.embed(x)*self.scale_factor


class SourceEmbedding(nn.Module):
    def __init__(self, src_vocab_size,d_model,dropout=0.1):
        super(SourceEmbedding,self).__init__()
        self.embed=Embeddings(src_vocab_size,d_model)
        self.positional_encoding=PositionEncoding(d_model,dropout)

    def forward(self,x):
        x=self.embed(x)
        return self.positional_encoding(x)

class TargetEmbedding(nn.Module):
    def __init__(self,tgt_vocab_size,d_model,dropout=0.1):

        super(TargetEmbedding,self).__init__()
        self.embed=Embeddings(tgt_vocab_size,d_model)
        self.positional_encoding=PositionEncoding(d_model,dropout)

    def forward(self,x):
        x=self.embed(x)
        return self.positional_encoding(x)

