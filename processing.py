import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import os
import datetime, time

def main(args):
    COMMENT_TEXT    = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    TOXIC           = data.Field(sequential=False, use_vocab=False)
    SEVERE_TOXIC    = data.Field(sequential=False, use_vocab=False)
    OBSCENE         = data.Field(sequential=False, use_vocab=False)
    THREAT          = data.Field(sequential=False, use_vocab=False)
    INSULT          = data.Field(sequential=False, use_vocab=False)
    IDENTITY_HATE   = data.Field(sequential=False, use_vocab=False)

    train_data, val_data, test_data = data.TabularDataset.splits(
            path='processed_data/', train='train.csv',
            validation='valid.csv', test='test.csv', format='csv',
            skip_header=True, fields=[('id', None), ('comment_text', COMMENT_TEXT), ('toxic', TOXIC), ('severe_toxic', SEVERE_TOXIC), ('obscene', OBSCENE), ('threat', THREAT), ('insult', INSULT), ('identity_hate', IDENTITY_HATE)])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    COMMENT_TEXT.build_vocab(train_data, val_data, test_data)

    COMMENT_TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = COMMENT_TEXT.vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline', help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)
    args = parser.parse_args()
    main(args)