import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import pandas as pd
import math

from embeddings import Embeddings
from position_encode import position_encode

class Transformer(nn.Module):
    def __init__(self,d_model,d_ff,n_head,max_seq_length,inp_vocab_size,out_vocab_size,n,source_padding,target_padding):
        super().__init__()
        self.source_padding=source_padding
        self.target_padding=target_padding

        self.encoder_embedding=Embeddings(inp_vocab_size,d_model)
        self.decoder_embedding=Embeddings(out_vocab_size,d_model)

        self.position=position_encode(d_model,max_seq_length)
        #self.out_position=position_encode(d_model,max_output_length)

        self.encoders=nn.ModuleList([encoder(d_model,n_head,d_ff) for _ in range(n)])
        self.decoders=nn.ModuleList([decoder(d_model,n_head,d_ff) for _ in range(n)])

        self.pre_softmax = nn.Linear(d_model, out_vocab_size)
        self.dropout = nn.Dropout(p=0.1)

    def masking(self, source_sequence, target_sequence):
        device = source_sequence.device

        source_mask = (source_sequence != self.source_padding).unsqueeze(1).unsqueeze(2).to(device)
        target_mask = (target_sequence != self.target_padding).unsqueeze(1).unsqueeze(3).to(device)

        sequence_length = target_sequence.size(1)
        ones = torch.ones(sequence_length, sequence_length, device=device)

        future_mask = torch.tril(ones, diagonal=0).bool()
        target_mask = target_mask & future_mask

        return source_mask, target_mask

    def forward(self,source_seq,target_seq):
        encoder_embedded = self.dropout(self.position(self.encoder_embedding(source_seq)))
        decoder_embedded = self.dropout(self.position(self.decoder_embedding(target_seq)))

        encoder_mask, decoder_mask = self.masking(source_seq, target_seq)

        encoder_output, decoder_output = encoder_embedded, decoder_embedded

        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, encoder_mask)

        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, decoder_mask,encoder_output, encoder_mask)

        logits = self.pre_softmax(decoder_output)

        return logits
