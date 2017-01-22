#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from sklearn import metrics

name = 'SNEDE0000ENP'

pos_scores = [float(line[:-1].split('\t')[1]) for line in open('%s_positive_lsgkm_test' % name)]
neg_scores = [float(line[:-1].split('\t')[1]) for line in open('%s_negative_lsgkm_test' % name)]

scores = pos_scores + neg_scores
labels = [1] * len(pos_scores) + [0] * len(neg_scores)

auroc = metrics.roc_auc_score(labels, scores)
auprc = metrics.average_precision_score(labels, scores)

print auroc, auprc
