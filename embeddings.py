import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import pandas as pd
import math

class Embeddings(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.embed1=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embed1(x)
