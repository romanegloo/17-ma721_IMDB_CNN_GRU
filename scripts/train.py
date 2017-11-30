#!/usr/bin/env python3
"""IMDB sentimental analysis training script."""

import argparse
import logging
import os
import time
from pathlib import PosixPath

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler

from IMDB import utils, CnnImdbSA, GruImdbSA, data

logger = logging.getLogger()


def add_train_args(parser):
    # Runtime environment
    runtime = parser.add_argument_group('Environment')

    runtime.add_argument('--num-epochs', type=int, default=5,
                         help='Train data iterations')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--batch-size', type=int, default=10,
                         help='batch size for training')
    runtime.add_argument('--data-workers', type=int, default=4,
                         help='Number of subprocesses for data loading')

    # files
    files = parser.add_argument_group('Files')
    files.add_argument('--data-dir', type=str,
                       help='Path to the directory containing test/train files')
    files.add_argument('--train-file', type=str,
                       help='Path to preprocessed train file')
    files.add_argument('--test-file', type=str,
                       help='Path to preprocessed test file')
    files.add_argument('--embedding-file', type=str,
                       help='path to Space-separated embeddings file')

    # saving + loading (continue from pretrained model)
    save_load = parser.add_argument_group('Files')
    save_load.add_argument('--checkpoint', action='store_true',
                           help='Save model + optimizer state after each epoch')


def add_model_args(parser):
    # Model architecture
    model = parser.add_argument_group('Model Architecture')
    model.add_argument('--model-type', type=str, default='cnn',
                       choices=['cnn', 'rnn'],
                       help='Model architecture type')
    model.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--hidden-dim', type=int, default=128,
                       help='GRU hidden dimension')
    model.add_argument('--doc-maxlen', type=int, default=500,
                       help='Max length of document in words')
    model.add_argument('--kernel-num', type=int, default=100,
                       help='number of each kind of kernel')
    model.add_argument('--kernel-sizes', type=str, default='3,4,5',
                       help='Comma-separated kernel sizes used for CNN')
    model.add_argument('--dropout', type=float, default=0.5,
                       help='the probability for dropout')
    model.add_argument('--weight-decay', type=float, default=0,
                       help='Weight decay factor for optimizer')

# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_model(args, train_exs, dev_exs):
    """Initializing a new model"""
    # Create a feature dict out of the annotations in the data
    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = utils.build_feature_dict(args, train_exs)
    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)

    # Build a dictionary from the contexts
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict(args, train_exs + dev_exs)
    logger.info('Num words = %d' % len(word_dict))
    args.vocab_size = len(word_dict)

    # Initialize model
    if args.model_type == 'cnn':
        model = CnnImdbSA(args, word_dict)
    else:
        model = GruImdbSA(args, word_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Validate
# ------------------------------------------------------------------------------

def validate(data_loader, model, stats, mode):
    logger.info('validateing {}...'.format(mode))
    examples = 0
    accuracies = []
    for ex in data_loader:
        batch_size = ex[0].size(0)
        # if mode == 'test':
        #     print(model.args.doc_maxlen, ex)
        predict = model.predict(ex)
        accuracy = ex[2].eq(predict.data).sum() / batch_size
        examples += batch_size
        accuracies.append(accuracy)
        if examples >= 5e3:
            break
    logger.info("{} validation: Epoch = {} | examples = {} | accuracy = {}"
                ''.format(mode, stats['epoch'], examples,
                          sum(accuracies)/len(accuracies)))
    return sum(accuracies) / len(accuracies)


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    logger.info('-' * 100)
    train_exs = utils.load_data(args, args.train_file)
    logger.info('{} train examples loaded'.format(len(train_exs)))
    test_exs = utils.load_data(args, args.test_file)
    logger.info('{} test examples loaded'.format(len(test_exs)))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint:
        raise NotImplementedError
    else:
        logger.info('Initializing a new model')
        model = init_model(args, train_exs, test_exs)

    # fix embeddings
    for p in model.embedding.parameters():
        p.requires_grad = False
    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adamax(parameters, weight_decay=args.weight_decay)

    # todo. using GPU?
    # todo. using multiple GPUs?

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = data.ImdbDataset(train_exs, model)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler.RandomSampler(train_dataset),
        num_workers=args.data_workers
    )
    test_dataset = data.ImdbDataset(test_exs, model)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=sampler.SequentialSampler(test_dataset)
    )

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'epoch': 0, 'best_valid': 0}
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # train update sequence
        model.train()
        losses = [[], 0]
        for idx, ex in enumerate(train_loader):
            # todo, transfer to GPU (args.use_cuda)
            inputs = [e if e is None else Variable(e) for e in ex[:2]]
            if args.model_type == 'cnn':
                logit = model(*inputs)
            else:
                logit = model(*inputs, model.initHidden(inputs[0].size()[0]))
            loss = F.cross_entropy(logit, Variable(ex[2]))

            # Clear gradients and run backward
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            # todo. check if this works
            # torch.nn.utils.clip_grad_norm(model.parameters(),
            #                               args.grad_clipping)

            # Update parameters
            optimizer.step()

            losses[0].append(loss.data[0])
            losses[1] += 1
            if idx % 5 == 0:
                logger.info('train: Epoch = {} | iter = {}/{} | loss = {:.2E}'
                            ''.format(epoch, idx, len(train_loader),
                                      sum(losses[0])/losses[1]))
                losses = [[], 0]
        logger.info('train: Epoch {} done.'.format(stats['epoch']))

        # Validate
        valid_tr = validate(train_loader, model, stats, mode='train')
        valid_ts = validate(test_loader, model, stats, mode='test')
        if valid_ts > stats['best_valid']:
            logger.info('BEST VALID: accuracy={:.2f} (epoch {})'
                        ''.format(valid_ts, stats['epoch']))
            # todo. save the model
            stats['best_valid'] = valid_ts

        # Save best valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'IMDB sentimental analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    add_model_args(parser)
    args = parser.parse_args()

    # Set defaults
    if args.data_dir is None:
        args.data_dir = os.path.join(
            PosixPath(__file__).absolute().parents[1].as_posix(),
            'data/processed-toy'
        )
    if args.train_file is None:
        args.train_file = os.path.join(
            args.data_dir, 'imdb-processed-train.txt' )
    if args.test_file is None:
        args.test_file = os.path.join(
            args.data_dir, 'imdb-processed-test.txt' )
    if args.embedding_file is None:
        args.embedding_file = os.path.join(
            PosixPath(__file__).absolute().parents[1].as_posix(),
            'data/embeddings/wiki.en.toy.vec'
        )
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.class_num = 2  # pos or neg


    # Set cuda
    # args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.random_seed)

    # Set logging - both to file and console
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    file = logging.FileHandler("run{}.log".format(int(time.time())))
    file.setFormatter(fmt)
    logger.addHandler(file)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Run!
    main(args)