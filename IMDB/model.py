#!/usr/bin/env python3
"""Network Wrapper"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _addindent
from torch.autograd import Variable
import numpy as np
import logging
import copy

from .network_cnn import ImdbCnn
from .network_rnn import ImdbGru

logger = logging.getLogger(__name__)


class ImdbSA(object):
    """high-level model that handles initializing the underlying network
    architecture saving, updating examples, and predicting examples."""

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, state_dict=None, word_dict=None,
                 feature_dict=None):
        # Book-keeping
        self.args = args
        self.num_free_params = 0
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        # Building network.
        if args.model_type == 'cnn':
            self.network = ImdbCnn(args)
        elif args.model_type == 'rnn':
            self.network = ImdbGru(args)
        else:
            raise RuntimeError('Unsupported Model Type %s' % args.model_type)

        # Load saved state
        if state_dict:
            self.network.load_state_dict(state_dict)

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.encoder.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            skip_first_line = True
            for line in f:
                if skip_first_line:
                    skip_first_line = False
                    continue
                parsed = line.rstrip().split(' ')
                if len(parsed) != embedding.size(1) + 1:
                    print(len(parsed), embedding.size)
                    print(line)
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network."""
        if self.args.fix_embeddings:
            for p in self.network.encoder.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # Transfer to CPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True))
                      for e in ex[:-1]]
            target = Variable(ex[-1].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:-1]]
            target = Variable(ex[-1])

        # Run forward
        if self.args.model_type == 'cnn':
            logit = self.network(*inputs)
        else:
            logit = self.network(*inputs,
                                 self.network.init_hidden(inputs[0].size()[0]))

        # Compute loss and accuracies
        loss = F.cross_entropy(logit, target)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)
        # Update parameters
        self.optimizer.step()
        self.updates += 1

        return loss.data[0], ex[0].size(0)

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else
                      Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:2]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True)
                      for e in ex[:2]]

        # Run forward
        if self.args.model_type == 'cnn':
            logits = self.network(*inputs)
        else:
            logits = self.network(*inputs,
                                  self.network.init_hidden(ex[0].size()[0]))


        # Decode predictions
        values, indices = logits.max(1)
        return indices


    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(selfself, filename, epoch):
        pass

    @staticmethod
    def load(filename):
        pass

    @staticmethod
    def load_checkpoint(filename):
        pass

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)

    # ------------------------------------------------------------------------------
    # Utils
    # - torch_summarize: displays the summary note with weights and parameters of
    #  the network (obtained from http://bit.ly/2glYWVV)
    # ------------------------------------------------------------------------------

    def torch_summarize(self, show_weights=True, show_parameters=True):
        """Summarizes torch model by showing trainable parameters and weights."""
        tmpstr = self.network.__class__.__name__ + ' (\n'
        total_params = 0
        total_params_wo_embedding = 0
        for key, module in self.network._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                nn.modules.container.Container,
                nn.modules.container.Sequential
            ]:
                modstr = self.torch_summarize(self.network)
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

        tmpstr += ')\ntotal_params: %d, total_params_wo_embedding: %d' % \
                  (total_params, total_params_wo_embedding)
        self.num_free_params = total_params_wo_embedding
        return tmpstr
