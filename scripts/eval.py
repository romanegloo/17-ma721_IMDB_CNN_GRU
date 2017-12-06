#!/usr/bin/env python3
"""interactive interface to perform sentimental analysis on arbitrary text"""
import torch
from torch.autograd import Variable
import argparse
import code
import logging
import spacy

from IMDB import CnnImdbSA

# initialize logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-file', type=str, help='Path to trained model')
parser.add_argument('--model-type', type=str, default='cnn',
                    choices=['cnn', 'rnn'], help='Model architecture type')
args = parser.parse_args()

args.doc_maxlen = 250

# load model
logger.info('loading trained model from {}'.format(args.model_file))
if args.model_type =='cnn':
    # reading model parameters
    saved_params = torch.load(
        args.model_file, map_location=lambda storage, loc: storage
    )
    saved_word_dict = saved_params['word_dict']
    model = CnnImdbSA(saved_params['args'],
                      saved_params['word_dict'],
                      saved_params['state_dict'])
    model.args.cuda = False  ## force to use CPU mode

ex = "This movie is full of entertaining."
# load tokenizer
nlp = spacy.load('en')


def process(review):
    # look up dictionary
    doc = [d for d in nlp(review.lower())]
    doc = [w.text if w not in saved_word_dict else '<UNK>' for w in doc]
    if len(doc) >= args.doc_maxlen:
        doc = doc[:args.doc_maxlen]
    else:
        doc.extend(['<NULL>'] * (args.doc_maxlen - len(doc)))
    doc = torch.LongTensor([saved_word_dict[w] for w in doc])

    logit = model.forward(Variable(doc.unsqueeze(0)), 0)
    _, index = logit.max(1)
    print(logit, 'pos' if index == 1 else 'neg')

banner = """
Sentimental Analysis on IMDb Movie Reviews
>> ex = "This movie is full of entertaining."
>> process(ex)
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())

