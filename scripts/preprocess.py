#!/usr/bin/env python3
"""Preprocess the IMDB dataset for training"""

import argparse
import json
import logging
import os
import re
import subprocess
import time
from multiprocessing import cpu_count
from pathlib import PosixPath

import spacy


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def load_dataset(path):
    """Load IMDB files and store fields separately."""
    output = {'exids': [], 'contexts': [], 'labels': [], 'scores': []}
    clean_re = re.compile('<.*?>')
    for subdir in ['neg', 'pos']:
        for f in os.listdir(os.path.join(path, subdir)):
            filename, ext = os.path.splitext(f)
            if ext != '.txt':
                continue
            (exid, score) = filename.split('_')
            output['exids'].append(exid)
            output['scores'].append(score)
            label = 0 if subdir == 'neg' else 1
            output['labels'].append(label)
            with open(os.path.join(path, subdir, f)) as in_f:
                context = in_f.read()
                context = re.sub(clean_re, ' ', context)
                context.replace('\n', ' ')
                output['contexts'].append(context)
    return output


def process_dataset(data, n_threads=None):
    """tokenize and parse the examples in multithread"""
    # initialize spacy tokenizer
    nlp = spacy.load('en')

    logging.info('Tokenizing examples...')
    docs = [d for d in nlp.pipe(data['contexts'], batch_size=10000,
                                n_threads=n_threads)]
    for idx in range(len(data['exids'])):
        yield {
            'id': data['exids'][idx],
            'context': [t.text.lower() for t in docs[idx]],
            'pos': [t.pos_ for t in docs[idx]],
            'ner': [t.ent_type_ for t in docs[idx]],
            'score': data['scores'][idx],
            'label': data['labels'][idx]
        }


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, help='Path to IMDB data directory')
parser.add_argument('--out-dir', type=str, help='Path to output file dir')
parser.add_argument('--n-threads', type=int, default=10)
args = parser.parse_args()

# initiate logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# set defaults
if args.data_dir is None:
    args.data_dir = os.path.join(
        PosixPath(__file__).absolute().parents[1].as_posix(),
        'data/aclImdb')
if args.out_dir is None:
    args.out_dir = os.path.join(
        PosixPath(__file__).absolute().parents[1].as_posix(),
        'data/processed')
if args.n_threads is None:
    args.n_threads = cpu_count()
t0 = time.time()

for mode in ['train', 'test']:
    path = os.path.join(args.data_dir, mode)
    logging.info('Loading {} dataset from {}'.format(mode, path))
    dataset = load_dataset(path)

    if not os.path.exists(args.out_dir):
        subprocess.call(['mkdir', '-p', args.out_dir])

    out_file = os.path.join(args.out_dir, 'imdb-processed-{}.txt'.format(mode))
    with open(out_file, 'w') as f:
        for ex in process_dataset(dataset, args.n_threads):
            f.write(json.dumps(ex) + '\n')
