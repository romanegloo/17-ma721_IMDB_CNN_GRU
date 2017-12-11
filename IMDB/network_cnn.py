#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImdbCnn(nn.Module):
    def __init__(self, args):
        super(ImdbCnn, self).__init__()
        self.args = args

        # define dimensions
        V = args.vocab_size
        E = args.embedding_dim
        C = args.class_num  # number of classes
        ConvIn = 1
        ConvOut = args.kernel_num
        Ks = args.kernel_sizes

        # define layers
        self.encoder = nn.Embedding(V, E, padding_idx=0)
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(ConvIn, ConvOut, (K, E)) for K in Ks]
        )
        self.dropout = nn.Dropout(args.dropout)
        self.decoder = nn.Linear(len(Ks) * ConvOut, C)

    def forward(self, x, x_len):
        x = self.encoder(x)      # (N, W, E)
        x = x.unsqueeze(1)      # (N, 1, W, E)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
                                # (N, Co, W-Ks_i) * len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
                                # (N, Co) * len(Ks)
        x = torch.cat(x, 1)     # (N, Co * len(Ks))
        x = self.dropout(x)
        logit = self.decoder(x)
        return logit
