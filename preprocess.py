#!/usr/bin/env python
# encoding: utf-8

import argparse

def seq2ngram(seqs, k, s, dest):
    f = open(seqs)
    lines = f.readlines()
    f.close()
    print 'need to n-gram %d lines' % len(lines)
    f = open(dest, 'w')
    for num, line in enumerate(lines):
        if num % 1000 == 0:
            print '%d lines to n-grams' % num
        line = line[:-1].lower() # remove '\n' and lower ACGT
        l = len(line) # length of line
        for i in range(0,l,s):
            if i+k >= l+1:
                break
            f.write(''.join(line[i:i+k]))
            f.write(' ')
        f.write('\n')
    f.close()

def forglove(f, dest):
    f = open(f)
    lines = f.readlines()
    f.close()
    print 'need to save %d lines' % len(lines)
    with open(dest, 'w') as f:
        for num, line in enumerate(lines):
            if num % 1000 == 0:
                print '%d lines saved' % num
            f.write(line[:-1])
            f.write('none ' * 5)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('description=kenize dna sequences to kmers')
    parser.add_argument('-k', dest='k', type=int, default=6, help='length of k-mer')
    parser.add_argument('-s', dest='s', type=int, default=2, help='stride when slicing k-mers')
    args = parser.parse_args()

    print 'Tokenize seqs...'
    name = 'SNEDE0000ENM'
    seqs = './data/%s_pos' % name
    f = './data/%s_pos_%dgram_%dstride' % (name, args.k, args.s)
    seq2ngram(seqs, args.k, args.s, f)
    seqs = './data/%s_neg' % name
    f = './data/%s_neg_%dgram_%dstride' % (name, args.k, args.s)
    seq2ngram(seqs, args.k, args.s, f)

    f_glove = './data/%s_%dgram_%dstride_oneline' % (name, args.k, args.s)
    forglove('./data/%s_pos_%dgram_%dstride' % (name, args.k, args.s), f_glove)

