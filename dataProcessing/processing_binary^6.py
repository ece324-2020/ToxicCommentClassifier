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
    COMMENT_TEXT_TOXIC          = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    COMMENT_TEXT_SEVERE_TOXIC   = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    COMMENT_TEXT_OBSCENE        = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    COMMENT_TEXT_THREAT         = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    COMMENT_TEXT_INSULT         = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    COMMENT_TEXT_IDENTITY_HATE  = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)

    TOXIC           = data.Field(sequential=False, use_vocab=False)
    SEVERE_TOXIC    = data.Field(sequential=False, use_vocab=False)
    OBSCENE         = data.Field(sequential=False, use_vocab=False)
    THREAT          = data.Field(sequential=False, use_vocab=False)
    INSULT          = data.Field(sequential=False, use_vocab=False)
    IDENTITY_HATE   = data.Field(sequential=False, use_vocab=False)

    train_data_toxic, val_data_toxic, test_data_toxic = data.TabularDataset.splits(path='binary_data^6/toxic', train='train.csv', validation='valid.csv', test='test.csv', format='csv', skip_header=True, fields=[('id', None), ('comment_text', COMMENT_TEXT_TOXIC), ('toxic', TOXIC), ('severe_toxic', None), ('obscene', None), ('threat', None), ('insult', None), ('identity_hate', None)])
    train_iter_toxic, val_iter_toxic, test_iter_toxic = data.BucketIterator.splits((train_data_toxic, val_data_toxic, test_data_toxic), batch_sizes=(args.batch_size, args.batch_size, args.batch_size), sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    COMMENT_TEXT_TOXIC.build_vocab(train_data_toxic, val_data_toxic, test_data_toxic)
    COMMENT_TEXT_TOXIC.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab_toxic = COMMENT_TEXT_TOXIC.vocab

    train_data_severe_toxic, val_data_severe_toxic, test_data_severe_toxic = data.TabularDataset.splits(path='binary_data^6/severe_toxic', train='train.csv', validation='valid.csv', test='test.csv', format='csv', skip_header=True, fields=[('id', None), ('comment_text', COMMENT_TEXT_SEVERE_TOXIC), ('toxic', None), ('severe_toxic', SEVERE_TOXIC), ('obscene', None), ('threat', None), ('insult', None), ('identity_hate', None)])
    train_iter_severe_toxic, val_iter_severe_toxic, test_iter_severe_toxic = data.BucketIterator.splits((train_data_severe_toxic, val_data_severe_toxic, test_data_severe_toxic), batch_sizes=(args.batch_size, args.batch_size, args.batch_size), sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    COMMENT_TEXT_SEVERE_TOXIC.build_vocab(train_data_severe_toxic, val_data_severe_toxic, test_data_severe_toxic)
    COMMENT_TEXT_SEVERE_TOXIC.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab_severe_toxic = COMMENT_TEXT_SEVERE_TOXIC.vocab

    train_data_obscene, val_data_obscene, test_data_obscene = data.TabularDataset.splits(path='binary_data^6/obscene', train='train.csv', validation='valid.csv', test='test.csv', format='csv', skip_header=True, fields=[('id', None), ('comment_text', COMMENT_TEXT_OBSCENE), ('toxic', None), ('severe_toxic', None), ('obscene', OBSCENE), ('threat', None), ('insult', None), ('identity_hate', None)])
    train_iter_obscene, val_iter_obscene, test_iter_obscene = data.BucketIterator.splits((train_data_obscene, val_data_obscene, test_data_obscene), batch_sizes=(args.batch_size, args.batch_size, args.batch_size), sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    COMMENT_TEXT_OBSCENE.build_vocab(train_data_obscene, val_data_obscene, test_data_obscene)
    COMMENT_TEXT_OBSCENE.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab_obscene = COMMENT_TEXT_OBSCENE.vocab

    train_data_threat, val_data_threat, test_data_threat = data.TabularDataset.splits(path='binary_data^6/', train='train.csv', validation='valid.csv', test='test.csv', format='csv', skip_header=True, fields=[('id', None), ('comment_text', COMMENT_TEXT_THREAT), ('toxic', None), ('severe_toxic', None), ('obscene', None), ('threat', THREAT), ('insult', None), ('identity_hate', None)])
    train_iter_threat, val_iter_threat, test_iter_threat = data.BucketIterator.splits((train_data_threat,threatval_data_threat, test_data_threat), batch_sizes=(args.batch_size, args.batch_size, args.batch_size), sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    COMMENT_TEXT_THREAT.build_vocab(train_data_threat, val_data_threat, test_data_threat)
    COMMENT_TEXT_THREAT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab_threat = COMMENT_TEXT_THREAT.vocab

    train_data_insult, val_data_insult, test_data_insult = data.TabularDataset.splits(path='binary_data^6/insult', train='train.csv', validation='valid.csv', test='test.csv', format='csv', skip_header=True, fields=[('id', None), ('comment_text', COMMENT_TEXT_INSULT), ('toxic', None), ('severe_toxic', None), ('obscene', None), ('threat', None), ('insult', INSULT), ('identity_hate', None)])
    train_it_insulter_insult, val_iter_insult, test_iter_insult = data.BucketIterator.splits((train_data_insult, val_data_insult, test_data_insult), batch_sizes=(args.batch_size, args.batch_size, args.batch_size), sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    COMMENT_TEXT_INSULT.build_vocab(train_data_insult, val_data_insult, test_data_insult)
    COMMENT_TEXT_INSULT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab_insult = COMMENT_TEXT_INSULT.vocab

    train_data_identity_hate, val_data_identity_hate, test_data_identity_hate = data.TabularDataset.splits(path='binary_data^6/identity_hate', train='train.csv', validation='valid.csv', test='test.csv', format='csv', skip_header=True, fields=[('id', None), ('comment_text', COMMENT_TEXT_IDENTITY_HATE), ('toxic', None), ('severe_toxic', None), ('obscene', None), ('threat', None), ('insult', None), ('identity_hate', IDENTITY_HATE)])
    train_iter_identity_hate, val_iter_identity_hate, test_iter_identity_hate = data.BucketIterator.splits((train_data_identity_hate, val_data_identity_hate, test_data_identity_hate), batch_sizes=(args.batch_size, args.batch_size, args.batch_size), sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    COMMENT_TEXT_IDENTITY_HATE.build_vocab(train_data_identity_hate, val_data_identity_hate, test_data_identity_hate)
    COMMENT_TEXT_IDENTITY_HATE.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab_identity_hate = COMMENT_TEXT_IDENTITY_HATE.vocab

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