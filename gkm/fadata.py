#!/usr/bin/env python
# encoding: utf-8

import numpy as np
np.random.seed(12345)

# name = 'SNEDE0000EJN'
names = ['SNEDE0000EMT', 'SNEDE0000EPC', 'SNEDE0000EPH', 'SNEDE0000ENO',
         'SNEDE0000EMU', 'SNEDE0000ENP']
i = 5
name = names[i]
k = 6
s = 2

pos_idx = [i for i, line in enumerate(open('../data/%s_pos_%dgram_%dstride' % (name, k, s))) if len(line.split())>15]
neg_idx = [i for i, line in enumerate(open('../data/%s_neg_%dgram_%dstride' % (name, k, s))) if len(line.split())>15]
pos_seqs = open('../data/%s_pos' % name).readlines()
neg_seqs = open('../data/%s_neg' % name).readlines()
pos_seqs = [pos_seqs[i][:-1] for i in pos_idx]
neg_seqs = [neg_seqs[i][:-1] for i in neg_idx]
X = np.array(pos_seqs + neg_seqs)
y = np.array([1] * len(pos_idx) + [0] * len(neg_idx))
print 'there are %d sequences' % len(y)
# print X, y

n_seqs = len(y)
indices = np.arange(n_seqs)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

n_tr = int(n_seqs * 0.85)
n_va = int(n_seqs * 0.05)
n_te = n_seqs - n_tr - n_va
X_train = X[:n_tr]
y_train = y[:n_tr]
X_test = X[-n_te:]
y_test = y[-n_te:]

f_pos_train = open('./data/%s_train_positive.fa' % name, 'w')
f_neg_train = open('./data/%s_train_negative.fa' % name, 'w')
f_pos_test = open('./data/%s_test_positive.fa' % name, 'w')
f_neg_test = open('./data/%s_test_negative.fa' % name, 'w')

print '%d train samples' % n_tr
for i in xrange(n_tr):
    if i % 1000 == 0:
        print '%d samples loaded' % i
    seq = X_train[i]
    label = y_train[i]
    if 'n' in seq or 'N' in seq:
        continue
    if len(seq) > 2047:
        seq = seq[:2047]
    if label == 1:
        f_pos_train.write('>pos%d\n' % i)
        f_pos_train.write('%s\n' % seq)
    else:
        f_neg_train.write('>neg%d\n' % i)
        f_neg_train.write('%s\n' % seq)

print '%d test samples' % n_te
for i in xrange(n_te):
    if i % 1000 == 0:
        print '%d samples loaded' % i
    seq = X_test[i]
    label = y_test[i]
    if 'n' in seq or 'N' in seq:
        continue
    if len(seq)>2047:
        seq = seq[:2047]
    if label == 1:
        f_pos_test.write('>pos%d\n' % i)
        f_pos_test.write('%s\n' % seq)
    else:
        f_neg_test.write('>neg%d\n' % i)
        f_neg_test.write('%s\n' % seq)

f_pos_train.close()
f_neg_train.close()
f_pos_test.close()
f_neg_test.close()

