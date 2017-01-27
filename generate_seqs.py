#!/usr/bin/env python
# encoding: utf-8

import hickle as hkl
import gzip
import numpy as np
from preprocess import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', dest='e', type=int, default=0)
args = parser.parse_args()
e = args.e

expnames = ['SNEDE0000EMT', 'SNEDE0000EPC', 'SNEDE0000EPH', 'SNEDE0000ENO', 'SNEDE0000EMU', 'SNEDE0000ENP']
name = expnames[e]
path = '/home/xumin/spot/%s' % name
genome = '../deepenhancer/temp/sequences.hkl'
seqpath1 = './data/%s_pos' % name
seqpath2 = './data/%s_neg' % name

print 'loading whole genome...'
genome = hkl.load(genome)
print 'loaded!'

f = gzip.open(path)
f_1 = open(seqpath1, 'w')
f_2 = open(seqpath2, 'w')
contents = f.readlines()
f.close()
print 'totally %d lines' % len(contents)
for i, line in enumerate(contents):
    if i % 1000 == 0:
        print '%d lines converted...' % i
    values = line.split()
    chrid, start, end = (values[0], int(values[1]), int(values[2]))
    f_1.write(genome[chrid][start:end])
    f_1.write('\n')
    l = end - start
    f_2.write(genome[chrid][start+l*5:end+l*5])
    f_2.write('\n')
f_1.close()
f_2.close()

k=6
s=2
seq2ngram('./data/%s_pos' % name, k, s, './data/%s_pos_%dgram_%dstride' % (name, k, s))
seq2ngram('./data/%s_neg' % name, k, s, './data/%s_neg_%dgram_%dstride' % (name, k, s))

forglove('./data/%s_pos_%dgram_%dstride' % (name,k,s),
         './data/%s_%dgram_%dstride_oneline' % (name,k,s))
