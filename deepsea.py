#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping

# import tensorflow as tf

from sklearn import metrics
np.random.seed(12345)

import argparse

parser = argparse.ArgumentParser(description='deepsea training')
parser.add_argument('-gpu', dest='gpu', type=int, default=1, help='using which gpu')
parser.add_argument('-k', dest='k', type=int, default=6, help='length of kmer')
parser.add_argument('-s', dest='s', type=int, default=2, help='stride of slicing')
parser.add_argument('-batchsize', dest='batchsize', type=int, default=2000, help='size of one batch')
parser.add_argument('-test', dest='test', action='store_true', default=False, help='only test step')
parser.add_argument('-i', dest='i', type=int, default=0)
args = parser.parse_args()

names = ['SNEDE0000EMT', 'SNEDE0000EPC', 'SNEDE0000EPH', 'SNEDE0000ENO',
         'SNEDE0000EMU', 'SNEDE0000ENP']
name = names[args.i]
print 'Loading seq data...'
pos_idx = [i for i, line in enumerate(open('./data/%s_pos_%dgram_%dstride' % (name, args.k, args.s))) if len(line.split())>15]
neg_idx = [i for i, line in enumerate(open('./data/%s_neg_%dgram_%dstride' % (name, args.k, args.s))) if len(line.split())>15]
pos_seqs = open('./data/%s_pos' % name).readlines()
pos_seqs = [' '.join(pos_seqs[i][:-1].lower()) for i in pos_idx]
neg_seqs = open('./data/%s_neg' % name).readlines()
neg_seqs = [' '.join(neg_seqs[i][:-1].lower()) for i in neg_idx]
seqs = pos_seqs + neg_seqs
lens = [len(line.split()) for line in seqs]
n_seqs = len(lens)
print 'there are %d sequences' % n_seqs
print '  sequence length statistics:'
print '  max ', np.max(lens)
print '  min ', np.min(lens)
print '  mean ', np.mean(lens)
print '  25% ', np.percentile(lens, 25)
print '  50% ', np.median(lens)
print '  75% ', np.percentile(lens, 75)
y = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))

print 'Tokenizing seqs...'
MAX_LEN = 1000
NB_WORDS = 5
tokenizer = Tokenizer(nb_words=NB_WORDS)
tokenizer.fit_on_texts(seqs)
sequences = tokenizer.texts_to_sequences(seqs)
X = pad_sequences(sequences, maxlen=MAX_LEN)

acgt_index = tokenizer.word_index
print 'Found %s unique tokens.' % len(acgt_index)

print 'Spliting train, valid, test parts...'
indices = np.arange(n_seqs)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

n_tr = int(n_seqs * 0.85)
n_va = int(n_seqs * 0.05)
n_te = n_seqs - n_tr - n_va
X_train = X[:n_tr]
y_train = y[:n_tr]
X_valid = X[n_tr:n_tr+n_va]
y_valid = y[n_tr:n_tr+n_va]
X_test = X[-n_te:]
y_test = y[-n_te:]

embedding_vector_length = 4
nb_words = min(NB_WORDS, len(acgt_index)) # kmer_index starting from 1
print('Building model...')
# with tf.device('/gpu:%d' % args.gpu):
while True:
    model = Sequential()
    print 'fix embedding layer with one-hot vectors'
    acgt2vec={'a': np.array([1, 0, 0, 0], dtype='float32'),
              'c': np.array([0, 1, 0, 1], dtype='float32'),
              'g': np.array([0, 0, 1, 0], dtype='float32'),
              't': np.array([0, 0, 0, 1], dtype='float32'),
              'n': np.array([0, 0, 0, 0], dtype='float32')}
    embedding_matrix = np.zeros((nb_words+1, embedding_vector_length))
    for acgt, i in acgt_index.items():
        if i > NB_WORDS:
            continue
        vector = acgt2vec.get(acgt)
        if vector is not None:
            embedding_matrix[i] = vector
    model.add(Embedding(nb_words+1,
                        embedding_vector_length,
                        weights=[embedding_matrix],
                        input_length=MAX_LEN,
                        trainable=False))
    model.add(Convolution1D(100, 20, activation='relu'))
    model.add(MaxPooling1D(13, 13))
    model.add(Dropout(0.2))
    model.add(Convolution1D(80, 8, activation='relu'))
    model.add(MaxPooling1D(4, 4))
    model.add(Dropout(0.2))
    model.add(Convolution1D(80, 8, activation='relu'))
    model.add(MaxPooling1D(2, 2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

    if not args.test:
        checkpointer = ModelCheckpoint(filepath="./model/%s_bestmodel_deepsea.hdf5"
                                       % name, verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=4, verbose=1)

        print 'Training model...'
        model.fit(X_train, y_train, nb_epoch=60, batch_size=args.batchsize, shuffle=True,
                  validation_data=(X_valid, y_valid),
                  callbacks=[checkpointer,earlystopper],
                  verbose=1)

    print 'Testing model...'
    model.load_weights('./model/%s_bestmodel_deepsea.hdf5'% name)
    tresults = model.evaluate(X_test, y_test, show_accuracy=True)
    print tresults
    y_pred = model.predict(X_test, batch_size=args.batchsize, verbose=1)
    y = y_test
    print 'Calculating AUC...'
    auroc = metrics.roc_auc_score(y, y_pred)
    auprc = metrics.average_precision_score(y, y_pred)
    print auroc, auprc
    break
