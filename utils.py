import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def idx2word2(idx, i2w, pad_idx):
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str

def idx2word(idx, i2w, pad_idx):
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[word_id.item()] + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps+2)

    return interpolation.T


def expierment_name(args, ts):
    exp_name = str()
    exp_name += "BS=%i_" % args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_" % args.embedding_size
    exp_name += "%s_" % args.rnn_type.upper()
    exp_name += "HS=%i_" % args.hidden_size
    exp_name += "L=%i_" % args.num_layers
    exp_name += "BI=%i_" % args.bidirectional
    exp_name += "LS=%i_" % args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_" % args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_" % args.x0
    exp_name += "TS=%s" % ts

    return exp_name

def partial_experiment_name(args, ts):
    exp_name = str()
    try:
        exp_name += "BS=%i_" % args.batch_size
    except AttributeError:
        pass
    try:
        exp_name += "LR={}_".format(args.learning_rate)
    except AttributeError:
        pass
    try:
        exp_name += "EB=%i_" % args.embedding_size
    except AttributeError:
        pass
    try:
        exp_name += "%s_" % args.rnn_type.upper()
    except AttributeError:
        pass
    try:
        exp_name += "HS=%i_" % args.hidden_size
    except AttributeError:
        pass
    try:
        exp_name += "L=%i_" % args.num_layers
    except AttributeError:
        pass
    try:
        exp_name += "BI=%i_" % args.bidirectional
    except AttributeError:
        pass
    try:
        exp_name += "LS=%i_" % args.latent_size
    except AttributeError:
        pass
    try:
        exp_name += "WD={}_".format(args.word_dropout)
    except AttributeError:
        pass
    try:
        exp_name += "ANN=%s_" % args.anneal_function.upper()
    except AttributeError:
        pass
    try:
        exp_name += "K={}_".format(args.k)
    except AttributeError:
        pass
    try:
        exp_name += "X0=%i_" % args.x0
    except AttributeError:
        pass
    try:
        exp_name += "TS=%s" % ts
    except AttributeError:
        pass
    
    return exp_name
