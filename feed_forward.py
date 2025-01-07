import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import pandas as pd
import math

class feed_forward(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.relu=nn.ReLU()
        self.lin1=nn.Linear(d_model,d_ff)
        self.lin2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        x=self.lin1(x)
        x=self.relu(x)
        x=self.lin2(x)
        return x
