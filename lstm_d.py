#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn import metrics
np.random.seed(12345)

import argparse

parser = argparse.ArgumentParser(description='different dimension d')
parser.add_argument('-gpu', dest='gpu', type=int, default=5, help='using which gpu')
parser.add_argument('-d', dest='d', type=int, default=100, help='embedding dimension')
parser.add_argument('-k', dest='k', type=int, default=6, help='length of kmer')
parser.add_argument('-s', dest='s', type=int, default=2, help='stride of slicing')
parser.add_argument('-batchsize', dest='batchsize', type=int, default=2000, help='size of one batch')
parser.add_argument('-init', dest='init', action='store_true', default=True, help='initialize vector')
parser.add_argument('-noinit', dest='init', action='store_false', help='no initialize')
parser.add_argument('-trainable', dest='trainable', action='store_true', default=True, help='embedding vectors trainable')
parser.add_argument('-notrainable', dest='trainable', action='store_false', help='not trainable')
parser.add_argument('-test', dest='test', action='store_true', default=False, help='only test step')
parser.add_argument('-i', dest='i', type=int, default=0, help='which dataset to use')
args = parser.parse_args()

names = ['SNEDE0000EMT', 'SNEDE0000EPC', 'SNEDE0000EPH', 'SNEDE0000ENO', 'SNEDE0000EMU', 'SNEDE0000ENP']
name = names[args.i]
print 'Loading seq data...'
pos_seqs = [line[:-2] for line in open('./data/%s_pos_%dgram_%dstride' % (name, args.k, args.s)) if len(line.split())>15]
neg_seqs = [line[:-2] for line in open('./data/%s_neg_%dgram_%dstride' % (name, args.k, args.s)) if len(line.split())>15]
seqs = pos_seqs + neg_seqs
lens = [len(line.split()) for line in seqs]
n_seqs = len(lens)
print 'there are %d sequences' % n_seqs
print '  containing %d-mers statistics:' % args.k
print '  max ', np.max(lens)
print '  min ', np.min(lens)
print '  mean ', np.mean(lens)
print '  25% ', np.percentile(lens, 25)
print '  50% ', np.median(lens)
print '  75% ', np.percentile(lens, 75)
y = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))

print 'Tokenizing seqs...'
MAX_LEN = 1000
NB_WORDS = 20000
tokenizer = Tokenizer(nb_words=NB_WORDS)
tokenizer.fit_on_texts(seqs)
sequences = tokenizer.texts_to_sequences(seqs)
X = pad_sequences(sequences, maxlen=MAX_LEN)

kmer_index = tokenizer.word_index
print 'Found %s unique tokens.' % len(kmer_index)

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

embedding_vector_length = args.d
nb_words = min(NB_WORDS, len(kmer_index)) # kmer_index starting from 1
print('Building model...')
model = Sequential()
if args.init:
    print 'initialize embedding layer with glove vectors'
    kmer2vec={}
    f = open('./data/%s_%dgram_%dstride_%dD_vectors.txt' % (name, args.k, args.s, args.d))
    for line in f:
        values = line.split()
        try:
            kmer = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            kmer2vec[kmer] = coefs
        except:pass
    f.close()
    embedding_matrix = np.zeros((nb_words+1, embedding_vector_length))
    for kmer, i in kmer_index.items():
        if i > NB_WORDS:
            continue
        vector = kmer2vec.get(kmer)
        if vector is not None:
            embedding_matrix[i] = vector

    print 'embedding layers trainable %s' % args.trainable
    model.add(Embedding(nb_words+1,
                        embedding_vector_length,
                        weights=[embedding_matrix],
                        input_length=MAX_LEN,
                        trainable=args.trainable))
else:
    model.add(Embedding(nb_words+1,
                        embedding_vector_length,
                        input_length=MAX_LEN))
model.add(Dropout(0.2))
model.add(Convolution1D(100, 10, activation='relu'))
model.add(MaxPooling1D(4, 4))
model.add(Dropout(0.2))
model.add(Convolution1D(100, 8, activation='relu'))
model.add(MaxPooling1D(2, 2))
model.add(Dropout(0.2))
model.add(Convolution1D(80, 8, activation='relu'))
model.add(MaxPooling1D(2, 2))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(80, consume_less='gpu')))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

if not args.test:
    checkpointer = ModelCheckpoint(filepath="./model/%s_bestmodel_%dgram_%dstride_%dD_%sinit_%strainable.hdf5"
                                   % (name, args.k, args.s, args.d, args.init, args.trainable), verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    print 'Training model...'
    model.fit(X_train, y_train, nb_epoch=60, batch_size=args.batchsize, shuffle=True,
              validation_data=(X_valid, y_valid),
              callbacks=[checkpointer,earlystopper],
              verbose=1)

print 'Testing model...'
model.load_weights('./model/%s_bestmodel_%dgram_%dstride_%dD_%sinit_%strainable.hdf5'
                   % (name, args.k, args.s, args.d, args.init, args.trainable))
tresults = model.evaluate(X_test, y_test, show_accuracy=True)
print tresults
y_pred = model.predict(X_test, batch_size=args.batchsize, verbose=1)
y = y_test
print 'Calculating AUC...'
auroc = metrics.roc_auc_score(y, y_pred)
auprc = metrics.average_precision_score(y, y_pred)
print auroc, auprc
