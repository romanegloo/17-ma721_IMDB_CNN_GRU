#!/usr/bin/env python3
"""Data structures / helpers"""
import unicodedata
import torch
from torch.utils.data import Dataset

# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens

# ------------------------------------------------------------------------------
# PyTorch Dataset class for IMDB data
# ------------------------------------------------------------------------------


class ImdbDataset(Dataset):
    def __init__(self, examples, model):
        self.ex = examples
        self.w_dict = model.word_dict
        self.doc_maxlen = model.args.doc_maxlen
        self.lengths = [0] * len(examples)
        self.use_ft = model.args.use_feature_tags
        self.ft_dict = model.feature_dict if self.use_ft else None

        self._get_doc_lengths()

    def __len__(self):
        return len(self.ex)

    def _get_doc_lengths(self):
        for idx, ex in enumerate(self.ex):
            length = len(ex['context']) \
                if len(ex['context']) <= self.doc_maxlen else self.doc_maxlen
            self.lengths[idx] = length

    def __getitem__(self, idx):
        document = self.ex[idx]['context'][:]
        ft_ner = self.ex[idx]['ner'][:]
        ft_pos = self.ex[idx]['pos'][:]

        # padding the context of given example
        if self.lengths[idx] >= self.doc_maxlen:
            document = document[:self.doc_maxlen]
            ft_ner = ft_ner[:self.doc_maxlen]
            ft_pos = ft_pos[:self.doc_maxlen]
        else:
            document.extend(['<NULL>'] * (self.doc_maxlen - self.lengths[idx]))
            ft_ner.extend(['<NULL>'] * (self.doc_maxlen - self.lengths[idx]))
            ft_pos.extend(['<NULL>'] * (self.doc_maxlen - self.lengths[idx]))
        # index words
        document = \
            torch.LongTensor([self.w_dict[w] for w in document])

        # features (ner, pos)
        if self.use_ft:
            ft_ner = torch.LongTensor(
                [self.ft_dict['ner='+w] if 'ner='+w in self.ft_dict else -1
                 for w in ft_ner])
            ft_pos = torch.LongTensor(
                [self.ft_dict['pos='+w] if 'pos='+w in self.ft_dict else -1
                 for w in ft_pos])
            return document, self.lengths[idx], ft_ner, ft_pos, \
                   self.ex[idx]['label']
        else:
            return document, self.lengths[idx], self.ex[idx]['label']
