import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import pandas as pd
import math

class position_encode(nn.Module):
    def __init__(self,d_model,seq_length):
        super().__init__()
        pd=torch.zeros(seq_length,d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)        
        scaler = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term= torch.exp(-scaler* (math.log(10000) / d_model))
        pd[:, 0::2] = torch.sin(position * div_term)
        pd[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pd", pd.unsqueeze(0))

    def forward(self,x):
        return x + self.pd[:, :x.size(1)]
        
