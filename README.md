# ismb2017_lstm

## Introduction

This is the source code for paper

``Chromatin Accessibility Prediction via Convolutional Long Short-Term Memory Networks with k-mer Embedding'', Xu Min, Wanwen Zeng, Ning Chen, Ting Chen and Rui Jiang. Accepted by ISMB/ECCB 2017.

In this work, we address the problem of predicting chromatin accessibility from merely sequence information, by proposing an innovative convolutional long short-term memory network with k-mer embedding.

## Citation

Xu Min, Wanwen Zeng, Ning Chen, Ting Chen, Rui Jiang; Chromatin accessibility prediction via convolutional long short-term memory networks with k-mer embedding, *Bioinformatics*, Volume 33, Issue 14, 15 July 2017, Pages i92–i101, https://doi.org/10.1093/bioinformatics/btx234

## Install
The code is mainly written in Python (2.7) using Keras (1.1.0) with Theano backend. One can install the required modules by following instructions on website https://keras.io/#installation. 

The [Anaconda](https://www.continuum.io/downloads) platform is highly recommended. 

## Usage
### Preprocessing
First, we generate the sequence dataset and prepare k-mer corpus for GloVe.

```python
python ./generate_seqs.py -e 0
```
The $$k$$-mer length $$k$$, the splitting stride $$s$$ can be set in the script. The -e paramter assigns one of the six cell type: GM12878, K562, MCF-7, HeLa-S3, H1-hESC and HepG2.

### GloVe pre-training
We train k-mer embedding vectors by GloVe.

```bash
./demo.sh
```

### Supervised learning
We train the supervised deep learning model based on the datasets and the pre-trained k-mer vectors.

```python
THEANO_FLAGS='device=gpu0' python lstm.py -i 0 -batchsize 3000
```
One can find the meaning of each parameter in the Python script.
Other models, including the DeepSEA baseling model and some variant deep learning structures, are used in a similar way.
Best trained models will be saved in hdf5 files.
