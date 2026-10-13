# 多头注意力机制（使用论文中的变量名称）

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiAttentionHead(nn.Module):
    def __init__(self,d_model,h):
        super(MultiAttentionHead,self).__init__()
        assert d_model%h==0

        self.d_model=d_model
        self.h=h

        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.fc_out=nn.Linear(d_model,d_model)

    def forward(self,q,k,v,mask=None):
        batch_size=q.size(0)

        seq_len_q=q.size(1)
        seq_len_k=k.size(1)
        
        Q=self.w_q(q).view(batch_size,seq_len_q,self.h,-1).transpose(1,2)
        K=self.w_k(k).view(batch_size,seq_len_k,self.h,-1).transpose(1,2)
        V=self.w_v(v).view(batch_size,seq_len_k,self.h,-1).transpose(1,2)

        scaled_attention,_=self.scaled_dot_product_attention(Q,K,V,mask)

        concat_out=scaled_attention.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        out=self.fc_out(concat_out)
        return out
    

    def scaled_dot_product_attention(self,Q,K,V,mask=None):
        d_k=Q.size(-1)

        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            scores=scores.masked_fill(mask==0,float('-inf'))
        attentino_weights=F.softmax(scores,dim=-1)

        output=torch.matmul(attentino_weights,V)

        return output,attentino_weights

    
