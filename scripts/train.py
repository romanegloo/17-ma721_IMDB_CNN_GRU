#!/usr/bin/env python3
"""training a model"""

import argparse
import logging
import os
import sys
from multiprocessing import cpu_count
import copy

import torch
from torch.utils.data import DataLoader, sampler

from IMDB import utils, data
from IMDB import ImdbSA

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def add_arguments(parser):
    """define parameters with the user provided arguments"""
    parser.register('type', 'bool', str2bool)

    # Runtime Environment
    runtime = parser.add_argument_group('Runtime Environments')
    runtime.add_argument('--run-name', type=str,
                         help='Identifiable name for each run')
    runtime.add_argument('--random-seed', type=int,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--no-cuda', action='store_true',
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help="Specify GPU device id to use")
    runtime.add_argument('--num-epochs', type=int, default=30,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=64,
                         help='batch size for training')
    runtime.add_argument('--print-parameters', action='store_true',
                         help='Print out model parameters')
    runtime.add_argument('--save-plots', action='store_true',
                         help='Save plot files of losses/accuracies/etc.')

    # Files: set the paths to important files
    files = parser.add_argument_group('Files')
    files.add_argument('--data-dir', type=str,
                       help='Path to the directory containing test/train files')
    files.add_argument('--var-dir', type=str,
                       help='Path to var directory; log files stored')
    files.add_argument('--train-file', type=str,
                       help='Path to preprocessed train data file')
    files.add_argument('--test-file', type=str,
                       help='Path to preprocessed test data file')
    files.add_argument('--embedding-file', type=str,
                       help='path to space-separated embeddings file')

    # Model Architecture: model specific options
    model = parser.add_argument_group('Model Architecture')
    model.add_argument('--model-type', type=str, default='cnn',
                       choices=['cnn', 'rnn'],
                       help='Model architecture type')
    model.add_argument('--use-feature-tags', action='store_true',
                       help='add nlp annotations to the inputs')
    model.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--hidden-dim', type=int, default=64,
                       help='GRU hidden dimension')
    model.add_argument('--doc-maxlen', type=int, default=500,
                       help='Max length of document in words')
    model.add_argument('--kernel-num', type=int, default=64,
                       help='number of each kind of kernel')
    model.add_argument('--kernel-sizes', type=str, default='3,4,5',
                       help='Comma-separated kernel sizes used for CNN')
    model.add_argument('--dropout', type=float, default=0.5,
                       help='the probability for dropout')
    model.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay factor for optimizer')

    # Optimization details
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--optimizer', type=str, default='adamax',
                       help='Optimizer: sgd or adamax')
    optim.add_argument('--fix-embeddings', type='bool', default=True,
                       help='Keep word embeddings fixed (use pretrained)')
    optim.add_argument('--grad-clipping', type=float, default=10,
                       help='Gradient clipping')

    # Saving + Loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', action='store_true',
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')

def init():
    """set default values and initialize components"""
    # initialize logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)


    logger.info('-' * 100)
    logger.info('Initializing...')

    # set default values
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.class_num = 2  # pos or neg
    if args.run_name is None:
        import uuid
        import time
        args.run_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
    logger.info('RUN: {}'.format(args.run_name))
    args.data_workers = cpu_count()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set paths
    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(__file__),
                                     '../data')
    if args.var_dir is None:
        args.var_dir = os.path.join(os.path.dirname(__file__), '../var')
    if args.train_file is None:
        args.train_file = os.path.join(args.data_dir,
                                       'processed/imdb-processed-train.txt')
    else:
        args.train_file = os.path.join(args.data_dir, args.train_file)
    if args.test_file is None:
        args.test_file = os.path.join(args.data_dir,
                                      'processed/imdb-processed-test.txt')
    else:
        args.test_file = os.path.join(args.data_dir, args.test_file)

    if args.embedding_file is None:
        args.embedding_file = os.path.join(args.data_dir,
                                           'embeddings/wiki.en.vec')
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)
    # path to save model file
    args.model_file = os.path.join(args.var_dir,
                                   '{}-best.mdl'.format(args.run_name))

    # add file log handle (args.var_dir and args.run_name need to be defined)
    log_path = os.path.join(args.var_dir, 'run{}.log'.format(args.run_name))
    file = logging.FileHandler(log_path)
    file.setFormatter(fmt)
    logger.addHandler(file)

    # Set random state
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        # np.random.seed(args.random_seed)  # if numpy is used
        if args.cuda:
            torch.cuda.manual_seed(args.random_seed)


def prepare_dataloader():
    """Make data loaders for train and dev"""
    global args
    logger.info('-' * 100)
    logger.info('Loading Datasets...')
    train_ex = utils.load_data(args.train_file)
    logger.info('{} train examples loaded'.format(len(train_ex)))
    test_ex = utils.load_data(args.test_file)
    logger.info('{} test examples loaded'.format(len(test_ex)))

    logger.info('Building feature dictionary...')
    feature_dict = utils.build_feature_dict(train_ex)
    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)

    logger.info('Build word dictionary...')
    word_dict = utils.build_word_dict(train_ex + test_ex)
    logger.info('Num words = %d' % len(word_dict))
    args.vocab_size = len(word_dict)

    logger.info('-' * 100)
    logger.info('Creating DataLoaders')
    if args.cuda:
        kwargs = {'num_workers': 0, 'pin_memory': True}
    else:
        kwargs = {'num_workers': args.data_workers}
    train_dataset = data.ImdbDataset(args, train_ex, word_dict, feature_dict)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler.RandomSampler(train_dataset),
        **kwargs
    )
    test_dataset = data.ImdbDataset(args, test_ex, word_dict, feature_dict)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=sampler.RandomSampler(test_dataset),
        **kwargs
    )
    return train_loader, test_loader, word_dict, feature_dict

# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------

def train(args, data_loader, model, global_stats):
    logger.info('-' * 100)
    logger.info('Starting training/validation loop...')

    # Initialize meters and timers
    train_loss = utils.AverageMeter()
    train_loss_total = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        loss = model.update(ex)[0]
        train_loss.update(loss)
        train_loss_total.update(loss)

        if idx % 10 == 0:
            logger.info('train: Epoch = {} | iter = {}/{} | loss = {:.2E} |'
                        ' Elapsed time = {:.2f}'
                        ''.format(global_stats['epoch'], idx, len(data_loader),
                                  train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()
    global_stats['losses'].append(train_loss_total.avg)


def validate(args, data_loader, model, global_stats, mode):
    """Run one full validation"""
    epoch = global_stats['epoch']
    eval_time = utils.Timer()
    acc = utils.AverageMeter()
    best_updated = False

    examples = 0
    for ex in data_loader:
        batch_size = ex[0].size(0)
        pred = model.predict(ex)
        acc_ =  ex[-1].eq(pred.data.cpu()).sum() / batch_size
        acc.update(acc_, batch_size)

        # If getting train accuracies, sample max 10k
        examples += batch_size
        # if mode == 'train' and examples >= 1e4:
        if examples >= 1e3:
            break

    logger.info("{} validation: examples = {} | accuracy = {}"
                ''.format(mode, examples, acc.avg))
    if mode =='train':
        global_stats['acc_train'].append(acc.avg)
    else:
        if acc.avg > global_stats['best_valid']:
            global_stats['best_valid'] = acc.avg
            global_stats['best_valid_at'] = epoch
            best_updated = True
        global_stats['acc_test'].append(acc.avg)
        ratio = acc.avg / (model.num_free_params * (epoch + 1))**.1
        if ratio > global_stats['best_ratio']:
            logger.info('BEST RATIO: ratio={:.2f} (epoch {})'
                        ''.format(ratio, epoch))
            global_stats['best_ratio'] = ratio
            global_stats['best_ratio_at'] = epoch
    return best_updated


def report(stats):
    logger.info('-' * 100)
    logger.info('Report - RUN: {}'.format(args.run_name))
    logger.info('Best Valid: {} (epoch {}), Best Ratio: {} (epoch {})'
                ''.format(stats['best_valid'], stats['best_valid_at'],
                          stats['best_ratio'], stats['best_ratio_at']))
    if args.save_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x = list(range(args.num_epochs))
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
        out_file = os.path.join(args.var_dir,
                                'plot-{}.png'.format(args.run_name))
        plt.savefig(out_file)


if __name__ == '__main__':
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    # --------------------------------------------------------------------------
    # Arguments
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        'package_name',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Initialization and DataLoaders
    # --------------------------------------------------------------------------
    init()
    train_loader, test_loader, word_dict, feature_dict = prepare_dataloader()

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------
    model = None
    if args.checkpoint:
        # resume training
        pass
    else:
        # either use pretrained or newly initialized
        if args.pretrained:
            pass
        else:
            model = ImdbSA(args, word_dict=word_dict, feature_dict=feature_dict)

    model_summary = model.torch_summarize()
    if args.print_parameters:
        logger.info(model_summary)

    # set cpu/gpu mode
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
        model.cuda()
    else:
        logger.info('Running on CPU only.')
    # Use multiple GPUs?
    # if args.parallel:
    #     model.parallelize()

    model.init_optimizer()
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    # --------------------------------------------------------------------------
    # Train/Validation loop
    # --------------------------------------------------------------------------
    stats = {'epoch': 0, 'timer': utils.Timer(),
             'best_valid': 0, 'best_valid_at': 0,
             'best_ratio': 0, 'best_ratio_at': 0,
             'acc_train': [], 'acc_test': [], 'losses': []}

    for epoch in range(0, args.num_epochs):
        stats['epoch'] = epoch
        try:
            train(args, train_loader, model, stats)
            validate(args, train_loader, model, stats, mode='train')
            best_updated = \
                validate(args, test_loader, model, stats, mode='test')
            if best_updated:
                # save the best model
                params = {
                    'word_dict': model.word_dict,
                    'args': model.args,
                    'state_dict': copy.copy(model.network.state_dict())
                }
                try:
                    torch.save(params, args.model_file)
                except BaseException:
                    logger.warning('WARN: Saving failed... continuing anyway.')

        except KeyboardInterrupt:
            logger.warning('Training loop terminated')
            report(stats)
            exit(1)

    # --------------------------------------------------------------------------
    # Report the results
    # --------------------------------------------------------------------------
    report(stats)
