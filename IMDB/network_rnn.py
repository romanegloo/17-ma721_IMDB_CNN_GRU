#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable

class ImdbGru(nn.Module):
    def __init__(self, args):
        super(ImdbGru, self).__init__()
        self.args = args

        # define dimensions
        H = args.hidden_dim
        self.hidden_dim = H
        C = args.class_num
        V = args.vocab_size
        E = args.embedding_dim
        W = args.doc_maxlen

        # define layers
        self.encoder = nn.Embedding(V, E, padding_idx=0)
        self.gru = nn.GRU(input_size=E, hidden_size=H)
        self.decoder = nn.Linear(H, C)

    def forward(self, x, x_len, hidden):
        # compute sorted sequence lengths
        sorted_len, idx_sort = torch.sort(x_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = sorted_len.data.tolist() if type(sorted_len) == Variable \
            else sorted_len.tolist()

        # sort x
        x = x.index_select(0, idx_sort)

        x = self.encoder(x)           # (N, W, E)
        x = x.permute(1, 0, 2)

        # pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        x, hn = self.gru(rnn_input, hidden)      # x (1, W, H)
        x = hn.squeeze()                 # (V, H)
        logit = self.decoder(x)

        # unsort and return
        logit = logit.index_select(0, idx_unsort)
        return logit

    def init_hidden(self, N):
        if self.args.cuda:
            return Variable(torch.randn(1, N, self.hidden_dim)).cuda()
        else:
            return Variable(torch.randn(1, N, self.hidden_dim))
