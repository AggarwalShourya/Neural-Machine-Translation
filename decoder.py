import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import pandas as pd
import math

from multi_head import multi_head
from feed_forward import feed_forward

class decoder(nn.Module):
    def __init__(self,d_model,n_head,d_ff):
        super().__init__()
        self.attent=multi_head(d_model,n_head)
        #self.encode=encoder(d_model,n_head,d_ff)
        self.feeder=feed_forward(d_model,d_ff)
        self.norm=nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(p=0.1)

    def forward(self,x,mask,encode_out,encode_mask):
        output=self.dropout(self.attent(x,x,x,mask))
        x=self.norm(output+x)
        output=self.attent(x,encode_out,encode_out,encode_mask)
        output=self.dropout(output)
        x=self.norm2(output+x)
        output=self.feeder(x)
        x=self.norm3(output+x)

        return x
        
