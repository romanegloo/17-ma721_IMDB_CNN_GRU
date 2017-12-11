#!/usr/bin/env python3
"""Download datasets from external resources and pre-process"""
import os
import tarfile
from tqdm import tqdm
import requests

import re
import json
import spacy
from multiprocessing import cpu_count

# paths of urls/local directories
url_imdb = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
url_embedding = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec"
data_dir = os.path.join(os.path.dirname(__file__), '../data/')
dataset_file = os.path.basename(url_imdb)
embedding_file = os.path.basename(url_embedding)


def download():
    """download dataset and word embeddings, if not exists"""
    # dataset files
    outfile = os.path.join(data_dir, dataset_file)
    if not os.path.exists(outfile):
        _download(url_imdb, outfile)
        print("extracting dataset file...")
        tar = tarfile.open(outfile)
        tar.extractall(data_dir)
        tar.close()

    # word embedding file
    outfile = os.path.join(data_dir, embedding_file)
    if not os.path.exists(outfile):
        _download(url_embedding, outfile)


def _download(url, file):
    headread = requests.head(url)
    length = int(headread.headers['Content-length'])  # in bytes
    r = requests.get(url, stream=True)
    with open(file, 'wb') as f:
        pbar = tqdm(total=int(length/1024))
        for data in r.iter_content(chunk_size=1024):
            if data:
                pbar.update()
                f.write(data)


def preprocess():
    """preprocess data for NN model"""
    # read original data
    for mode in ['train', 'test']:
        path = os.path.join(data_dir, 'aclImdb', mode)
        print('Loading {} dataset from {}'.format(mode, path))
        dataset = _load_dataset(path)

        out_file = os.path.join(data_dir, 'imdb-processed-{}.txt'.format(mode))
        with open(out_file, 'w') as f:
            # transform data
            for ex in _process_dataset(dataset):
                # write the output
                f.write(json.dumps(ex) + '\n')


def _process_dataset(data, n_threads=cpu_count()):
    """tokenize and parse the examples in multithread"""
    # initialize spacy tokenizer
    nlp = spacy.load('en')

    print('Tokenizing examples...')
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


def _load_dataset(path):
    """Load IMDB files and store fields separately."""
    output = {'exids': [], 'contexts': [], 'labels': [], 'scores': []}
    clean_re = re.compile('<.*?>')  # remove html tags
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


if __name__ == '__main__':
    download()
    preprocess()