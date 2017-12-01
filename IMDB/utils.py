#!/usr/bin/env python3
"""Utilities for loading data and evaluating/reporting"""
import json
import logging
import numpy as np
import torch.nn as nn
from torch.nn.modules.module import _addindent

from .data import Dictionary

logger = logging.getLogger(__name__)


def load_data(args, filename):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(args.train_file) as f:
        examples = [json.loads(line) for line in f]
    return examples


def build_feature_dict(args, examples):
    """Index features (one hot) from fields in examples and options."""
    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}
    # Part of speech tag features
    for ex in examples:
        for w in ex['pos']:
            _insert('pos=%s' % w)

    # Named entity tag features
    for ex in examples:
        for w in ex['ner']:
            _insert('ner=%s' % w)

    return feature_dict


def build_word_dict(args, examples):
    """Return a dictionary from provided examples.
    """
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict


def load_words(args, examples):
    """Iterate and index all the words in examples (documents + questions)."""
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            words.add(w)

    words = set()
    for ex in examples:
        _insert(ex['context'])
    return words



# ------------------------------------------------------------------------------
# Utils
# - torch_summarize: displays the summary note with weights and parameters of
#  the network (obtained from http://bit.ly/2glYWVV)
# ------------------------------------------------------------------------------


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    total_params_wo_embedding = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            nn.modules.container.Container,
            nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        total_params += params
        if key != 'encoder':
            total_params_wo_embedding += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr, total_params, total_params_wo_embedding
