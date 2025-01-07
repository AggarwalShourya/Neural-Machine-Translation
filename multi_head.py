import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import pandas as pd
import math

class multi_head(nn.Module):
    def __init__(self,d_model,n_head):
        super().__init__()
        self.n_head=n_head
        self.d_model=d_model
        self.d_k=d_model//n_head
        self.q_lin=nn.Linear(d_model,d_model)
        self.k_lin=nn.Linear(d_model,d_model)
        self.v_lin=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)

    def splits(self,x):
        batch_size, seq_length, d_model = x.size()

        splits=x.view(batch_size,seq_length,self.n_head,self.d_k).transpose(1,2)
        return splits
    
    def attention(self,Q,K,V,mask=None):
        dot_attention=torch.matmul(Q,K.transpose(-1,-2))# Q--> (batch,head,seq_length,d_k)  K.transpose-->(batch,head,d_k,seq_length)
        dot_attention=dot_attention/math.sqrt(self.d_k)
        if mask is not None:
            dot_attention.masked_fill(mask==0,float('-inf'))
        
        logits_attention=torch.softmax(dot_attention,dim=-1)
        scaled_attention=torch.matmul(logits_attention,V)# V--> (batch,head,seq,d_k)

        return scaled_attention
    
    def concate(self,x):
        batch,self.n_head,seq_length,self.d_k=x.size()
        final_output=x.transpose(1,2)
        final_output=x.view(batch,seq_length,self.d_k*self.n_head)
        return final_output
    
    def forward(self,Q,K,V,mask=None):
        Q=self.q_lin(Q)
        K=self.k_lin(K)
        V=self.v_lin(V)

        Q=self.splits(Q)
        K=self.splits(K)
        V=self.splits(V)

        dot_attention=self.attention(Q,K,V,mask)

        output=self.concate(dot_attention)

        output=self.w_o(output)
        return output
        
        
