#!/usr/bin/env python3
"""Utilities for loading data and evaluating"""
import json
import logging

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
