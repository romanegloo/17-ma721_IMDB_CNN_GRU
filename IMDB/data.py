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
        # self.f_dict = model.feature_dict

        self._get_doc_lengths()

    def __len__(self):
        return len(self.ex)

    def _get_doc_lengths(self):
        for idx, ex in enumerate(self.ex):
            length = len(ex['context']) \
                if len(ex['context']) <= self.doc_maxlen else self.doc_maxlen
            self.lengths[idx] = length

    def __getitem__(self, idx):
        # padding the context of given example
        document = self.ex[idx]['context'][:]
        # length = len(context) if len(context) <= self.doc_maxlen else \
        #     self.doc_maxlen
        if self.lengths[idx] >= self.doc_maxlen:
            document = document[:self.doc_maxlen]
        else:
            document.extend(['<NULL>'] * (self.doc_maxlen - self.lengths[idx]))
        # index words
        document = \
            torch.LongTensor([self.w_dict[w] for w in document])

        return document, self.lengths[idx], self.ex[idx]['label']

