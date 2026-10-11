import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q,K,V,mask=None):
    
    embed_size=Q.size(-1)
    scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(embed_size)

    if mask is not None:
        scores=scores.masked_fill(mask==0,float('-inf'))

    attention_weight=F.softmax(scores,dim=-1)

    output=torch.matmul(attention_weight,V)

    return output,attention_weight



class Attention(nn.Module):
    def __init__(self, embed_size):

        super(Attention,self).__init__()
        self.embed_size=embed_size

        self.w_q=nn.Linear(embed_size,embed_size)
        self.w_k=nn.Linear(embed_size,embed_size)
        self.w_v=nn.Linear(embed_size,embed_size)

    def forward(self,q,k,v,mask=None):
        Q=self.w_q(q)
        K=self.w_k(k)
        V=self.w_v(v)

        out,attention_weight=scaled_dot_product_attention(Q,K,V,mask)
        return out,attention_weight
        

class SelfAttention(nn.Module):
    def __init__(self,embed_size):
        super(SelfAttention,self).__init__()

        self.attention=Attention(embed_size)

    def forward(self,x,mask=None):
        """
        q k v  come from the same x_seq
        """
        out,attention_weights=self.attention(x,x,x,mask)

        return out,attention_weights
        
       

class CrossAttention(nn.Module):
    def __init__(self,embed_size):
        super(CrossAttention,self).__init__()
        self.attention=Attention(embed_size)

    def forward(self,q,kv,mask=None):

        out,attentinon_weights=self.attention(q,kv,kv,mask)
        
        return out,attentinon_weights
    