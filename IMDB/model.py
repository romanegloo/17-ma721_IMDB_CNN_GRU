import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Convolutional Neural Network
# ------------------------------------------------------------------------------

class CnnImdbSA(nn.Module):
    def __init__(self, args, word_dict):
        super(CnnImdbSA, self).__init__()
        self.args = args
        self.num_parameters = 0
        self.word_dict = word_dict

        # dimensions
        V = args.vocab_size
        E = args.embedding_dim
        C = 2
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        logger.info("vocab_size: {}, embedding_dim: {}".format(V, E))

        self.encoder = nn.Embedding(V, E, padding_idx=0)
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(Ci, Co, (K, E)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.decoder = nn.Linear(len(Ks) * Co, C)

    def forward(self, x, x_len):
        if self.args.cuda:
            x = x.cuda()
        x = self.encoder(x)                        # (N, W, E)

        x = x.unsqueeze(1)                           # (N, Ci, W, E)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
                                                     # (N, Co, W-Ks_i) * len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
                                                     # (N, Co) * len(Ks)
        x = torch.cat(x, 1)                          # (N, Co * len(Ks))
        x = self.dropout(x)
        logit = self.decoder(x)
        logit = logit.cpu()
        return logit

    def predict(self, x):
        self.eval()
        logits = self.forward(x[0], x[1])
        values, indices = logits.max(1)
        return indices

    def load_embeddings(self, words, embedding_file):
        """Load pretrained word embeddings for a given list of words"""
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for {} words from {}'
                    ''.format(len(words), embedding_file))
        embedding = self.encoder.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            skip_first_line = True
            for line in f:
                if skip_first_line:
                    skip_first_line = False
                    continue
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1), \
                    'parsed: {}, embedding_dim: {}' \
                    ''.format(len(parsed), embedding.size(1))
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))


class CnnFeatImdbSA(nn.Module):
    def __init__(self, args, word_dict, feature_dict):
        super(CnnFeatImdbSA, self).__init__()
        self.args = args
        self.num_parameters = 0
        self.word_dict = word_dict
        self.feature_dict = feature_dict

        # dimensions
        V = args.vocab_size
        E = args.embedding_dim
        C = 2
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        logger.info("vocab_size: {}, embedding_dim: {}".format(V, E))

        self.encoder = nn.Embedding(V, E, padding_idx=0)
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(Ci, Co, (K, E)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.decoder = nn.Linear(len(Ks) * Co, C)

    def forward(self, x, x_len, ner, pos):
        if self.args.cuda:
            x = x.cuda()
        x = self.encoder(x)    # (N, W, E)
        # concatenate feature vector
        x = torch.cat((x, ner, pos), 2)

        x = x.unsqueeze(1)                           # (N, Ci, W, E)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
                                                     # (N, Co, W-Ks_i) * len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
                                                     # (N, Co) * len(Ks)
        x = torch.cat(x, 1)                          # (N, Co * len(Ks))
        x = self.dropout(x)
        logit = self.decoder(x)
        logit = logit.cpu()
        return logit

    def predict(self, x):
        self.eval()
        logits = self.forward(x[0], x[1])
        values, indices = logits.max(1)
        return indices

    def load_embeddings(self, words, embedding_file):
        """Load pretrained word embeddings for a given list of words"""
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for {} words from {}'
                    ''.format(len(words), embedding_file))
        embedding = self.encoder.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            skip_first_line = True
            for line in f:
                if skip_first_line:
                    skip_first_line = False
                    continue
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1), \
                    'parsed: {}, embedding_dim: {}' \
                    ''.format(len(parsed), embedding.size(1))
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

# ------------------------------------------------------------------------------
# RNN (Gated Recurrent Unit)
# ------------------------------------------------------------------------------

class GruImdbSA(nn.Module):
    def __init__(self, args, word_dict):
        super(GruImdbSA, self).__init__()
        self.args = args
        self.num_parameters = 0
        self.word_dict = word_dict

        # dimensions
        H = args.hidden_dim
        self.hidden_dim = args.hidden_dim
        C = args.class_num
        E = args.embedding_dim
        V = args.vocab_size
        W = args.doc_maxlen

        self.encoder = nn.Embedding(V, E)
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
        logit = logit.cpu()
        return logit

    def predict(self, x):
        self.eval()
        if self.args.cuda:
            inputs = [e if e is None else Variable(e.cuda())
                      for e in x[:2]]
        else:
            inputs = [e if e is None else Variable(e) for e in x[:2]]
        logits = self.forward(*inputs, self.initHidden(x[0].size()[0]))
        values, indices = logits.max(1)
        return indices

    def initHidden(self, N):
        if self.args.cuda:
            return Variable(torch.randn(1, N, self.hidden_dim)).cuda()
        else:
            return Variable(torch.randn(1, N, self.hidden_dim))

    def load_embeddings(self, words, embedding_file):
        """Load pretrained word embeddings for a given list of words"""
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for {} words from {}'
                    ''.format(len(words), embedding_file))
        embedding = self.encoder.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            skip_first_line = True
            for line in f:
                if skip_first_line:
                    skip_first_line = False
                    continue
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1), \
                    'parsed: {}, embedding_dim: {}' \
                    ''.format(len(parsed), embedding.size(1))
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))