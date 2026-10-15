import torch
import torch.nn as nn


# 前缀神经网络FNN

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
    def forward(self,x):
        return self.w_2(self.w_1(x).relu())


# 残差连接

class ResidualConnection(nn.Module):
    def __init__(self, dropout=0.1):
        super(ResidualConnection,self).__init__()
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(x))

# 层归一化  layer normalization

class LayerNorm(nn.Module):
    def __init__(self,feature_size,epsilon=1e-9):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(feature_size))
        self.beta=nn.Parameter(torch.zeros(feature_size))
        self.epsilon=epsilon

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        return self.gamma*(x-mean)/torch.sqrt(var+self.epsilon)+self.bata

# Add & Norm 模块

class SublayerConnection(nn.Module):
    def __init__(self,feature_size,dropout=0.1,epsilon=1e-9):
        super(SublayerConnection,self).__init__()
        self.residual=ResidualConnection(dropout)
        self.norm=LayerNorm(feature_size,epsilon)

    def forward(self,x,sublayer):
        return self.norm(self.residual(x,sublayer))

