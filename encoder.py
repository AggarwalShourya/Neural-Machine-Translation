import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import pandas as pd
import math

from feed_forward import feed_forward
from multi_head import multi_head

class encoder(nn.Module):
    def __init__(self,d_model,n_head,d_ff):
        super().__init__()
        self.attent=multi_head(d_model,n_head)
        self.feeder=feed_forward(d_model,d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(p=0.1)
    def forward(self,x,mask):
        x_t=self.dropout(self.attent(x,x,x,mask))
        x=self.norm1(x+x_t)
        x=self.feeder(x)
        x_l=self.dropout(x)
        x=self.norm2(x+x_l)

        return x
        
