import os
import numpy as np
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch import optim

class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim, padding_idx=word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data.copy_(word_vec_mat)

        
    def forward(self, inputs):

        return self.word_embedding(inputs)


class Encoder(nn.Module):
    def __init__(self, max_length, word_embedding_dim=50, hidden_size=230):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(max_length)


    def forward(self, inputs):
        return self.cnn(inputs)

    def cnn(self, inputs):
        x = self.conv(inputs.transpose(1, 2))
        x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(2) # n x hidden_size

class CNNSentenceEncoder(nn.Module):
    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50, 
            hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.word_vec_mat = word_vec_mat
        self.embedding = Embedding(word_vec_mat, max_length, word_embedding_dim)
        self.encoder = Encoder(max_length, word_embedding_dim, hidden_size)
        self.word2id = word2id

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        return x

    def tokenizer(self, tokens):
        all_tokens = []
        for raw_tokens in tokens:
            # token -> index
            indexed_tokens = []
            for token in raw_tokens:
                token = token.lower()
                if token in self.word2id:
                    indexed_tokens.append(self.word2id[token])
                else:
                    indexed_tokens.append(self.word2id['[UNK]']-1)

            len_it = len(indexed_tokens)
            if(len_it>self.max_length):
                len_it = self.max_length
    #         padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(self.word_vec_mat.shape[0]-1)
            indexed_tokens = indexed_tokens[:self.max_length]
            all_tokens.append(indexed_tokens)

        return all_tokens