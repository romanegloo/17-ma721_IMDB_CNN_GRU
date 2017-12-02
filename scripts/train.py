#!/usr/bin/env python3
"""IMDB sentimental analysis training script."""

import argparse
import logging
import os
import sys
import time
from pathlib import PosixPath

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler

from IMDB import utils, CnnImdbSA, CnnFeatImdbSA, GruImdbSA, data

logger = logging.getLogger()


def add_train_args(parser):
    # Runtime environment
    runtime = parser.add_argument_group('Environment')

    runtime.add_argument('--run-name', type=str,
                         help='identifiable name for each run')
    runtime.add_argument('--num-epochs', type=int, default=5,
                         help='Train data iterations')
    runtime.add_argument('--random-seed', type=int,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--batch-size', type=int, default=10,
                         help='batch size for training')
    runtime.add_argument('--data-workers', type=int, default=4,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--print-parameters', action='store_true',
                         help='Print out model parameters')
    runtime.add_argument('--save-plots', action='store_true',
                         help='Save plot files of losses/accuracies/etc.')
    runtime.add_argument('--no-cuda', action='store_true',
                         help='Use CPU only')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help="Specify GPU device id to use")

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
    model.add_argument('--use-feature-tags', action='store_true',
                       help='add nlp annotations to the inputs')
    model.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--hidden-dim', type=int, default=128,
                       help='GRU hidden dimension')
    model.add_argument('--doc-maxlen', type=int, default=500,
                       help='Max length of document in words')
    model.add_argument('--kernel-num', type=int, default=128,
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
    logger.info('Initializing a model (gpu?: {})'.format(args.cuda))
    if args.model_type == 'cnn':
        if args.cuda:
            if args.use_feature_tags:
                model = CnnFeatImdbSA(args, word_dict).cuda()
            else:
                model = CnnImdbSA(args, word_dict).cuda()
        else:
            if args.use_feature_tags:
                model = CnnFeatImdbSA(args, word_dict, feature_dict)
            else:
                model = CnnImdbSA(args, word_dict)
    else:
        if args.cuda:
            model = GruImdbSA(args, word_dict).cuda()
        else:
            model = GruImdbSA(args, word_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------

def train(model, train_loader, test_loader, optimizer, global_stats):
    for epoch in range(args.num_epochs):
        model.train()
        losses = [[], 0]
        losses_total = 0
        for idx, ex in enumerate(train_loader):
            if args.cuda:
                inputs = [e if e is None else Variable(e.cuda())
                          for e in ex[:-1]]
            else:
                inputs = [e if e is None else Variable(e) for e in ex[:-1]]
            if args.model_type == 'cnn':
                logit = model(*inputs)
            else:
                logit = model(*inputs, model.initHidden(inputs[0].size()[0]))
            loss = F.cross_entropy(logit, Variable(ex[-1]))

            # Clear gradients and run backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses[0].append(loss.data[0])
            losses[1] += 1
            losses_total += loss.data[0]

            if idx % 10 == 0:
                logger.info('train: Epoch = {} | iter = {}/{} | loss = {:.2E}'
                            ''.format(epoch, idx, len(train_loader),
                                      sum(losses[0])/losses[1]))
                losses = [[], 0]
        global_stats['losses'].append(
            round(losses_total / len(train_loader), 4))
        logger.info('train: Epoch {} done.'.format(epoch))

        # Validate
        valid_tr = validate(train_loader, model, global_stats, mode='train')
        global_stats['acc_train'].append(round(valid_tr, 4))
        valid_ts = validate(test_loader, model, global_stats, mode='test')
        global_stats['acc_test'].append(round(valid_ts, 4))
        ratio = valid_ts / (model.num_parameters * (epoch + 1))**.1
        global_stats['ratio'].append(ratio)
        if valid_ts > global_stats['best_valid']:
            logger.info('BEST VALID: accuracy={:.2f} (epoch {})'
                        ''.format(valid_ts, epoch))
            # todo. save the best model
            global_stats['best_valid'] = valid_ts
            global_stats['best_valid_at'] = epoch
        if ratio > global_stats['best_ratio']:
            logger.info('BEST RATIO: ratio={:.2f} (epoch {})'
                        ''.format(ratio, epoch))
            global_stats['best_ratio'] = ratio
            global_stats['best_ratio_at'] = epoch

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
        if examples >= 1e3:
            break
    logger.info("{} validation: examples = {} | accuracy = {}"
                ''.format(mode, examples, sum(accuracies)/len(accuracies)))
    return sum(accuracies) / len(accuracies)

# ------------------------------------------------------------------------------
# Save Plots
# ------------------------------------------------------------------------------


def save_plots(stats):
import matplotlib.pyplot as plt

# x = list(range(args.num_epochs))
# # losses
# plt.figure(figsize=(14, 8))
# plt.subplot(121)
# plt.plot(x, stats['losses'], 'r', label='train loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Train Losses')
# plt.legend()
#
#
# # accuracy
# plt.subplot(122)
# plt.plot(x, stats['acc_train'], 'g', label='train')
# plt.plot(x, stats['acc_test'], 'r', label='test')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.title('Test/Train Accuracies')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
# plt.savefig("plot-{}.png".format(args.run_name))

x = list(range(args['num_epochs']))
# losses
fig = plt.figure(figsize=(9, 4))
cnn = fig.add_subplot(121)
cnn.plot(x, stats['losses'], 'r', label='train loss')
cnn.set_xlabel('epoch')
cnn.set_ylabel('loss')
cnn.set_title('Train Losses')
cnn.legend()


# accuracy
rnn = fig.add_subplot(122)
rnn.plot(x, stats['acc_train'], 'g', label='train')
rnn.plot(x, stats['acc_test'], 'r', label='test')
rnn.set_ylim(ymax=1)
rnn.set_xlabel('epoch')
rnn.set_ylabel('accuracy')
rnn.set_title('Test/Train Accuracies')
rnn.legend(loc=4)

plt.tight_layout()
plt.show()
plt.savefig("plot-{}.png".format(args['run_name']))


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
    if args.checkpoint:
        raise NotImplementedError
    else:
        logger.info('Initializing a new model')
        model = init_model(args, train_exs, test_exs)

    # fix embeddings
    for p in model.encoder.parameters():
        p.requires_grad = False
    parameters = [p for p in model.parameters() if p.requires_grad]

    model_summary = utils.torch_summarize(model)
    if args.print_parameters:
         logger.info(model_summary)

    optimizer = torch.optim.Adamax(parameters, weight_decay=args.weight_decay)

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    logger.info('-' * 100)
    logger.info('Make data loaders')
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else \
        {'num_workers': args.data_workers}
    train_dataset = data.ImdbDataset(train_exs, model)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler.RandomSampler(train_dataset),
        **kwargs
    )
    test_dataset = data.ImdbDataset(test_exs, model)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=sampler.SequentialSampler(test_dataset),
        **kwargs
    )

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {
        'best_valid': 0,
        'best_valid_at': 0,
        'best_ratio': 0,
        'best_ratio_at': 0,
        'acc_train': [],
        'acc_test': [],
        'ratio': [],
        'losses': []
    }

    # train/validate loop
    try:
        train(model, train_loader, test_loader, optimizer, stats)
    except KeyboardInterrupt:
        logger.info(stats)
        if args.save_plots:
            save_plots(stats)
        exit(1)
    logger.info(stats)
    if args.save_plots:
        save_plots(stats)


if __name__ == '__main__':
    # Set logging - both to file and console
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

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
    if args.run_name is None:
        args.run_name = str(int(time.time()))
    logger.info('RUN: {}'.format(args.run_name))

    # add file log handle
    file = logging.FileHandler("run{}.log".format(args.run_name))
    file.setFormatter(fmt)
    logger.addHandler(file)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        # somehow, TITAN X is not working fine with my code. I had to
        # specifically set CUDA_VISIBLE_DEVICES to 1,2 and give --gpu 0 in
        # order to use GTX cards. don't know why.
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')

    # Set random state
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if args.cuda:
            torch.cuda.manual_seed(args.random_seed)

    # Run!
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    main(args)