#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
@desc:  
@author: TsungHan Yu  
@contact: nick.yu@hzn.com.tw  
@software: PyCharm  @since:python 3.6.0 on 2017/7/3
"""
import os

from gensim.models import word2vec
import re
import pandas as pd
import numpy as np
import itertools


rootdir = '../data/train_corpus_seg/'
dirs = os.listdir(rootdir)
dirs = [os.path.join(rootdir, f) for f in dirs]


# print(dirs)


def load_txt(x):
    with open(x) as f:
        res = [t for t in f]
        return ''.join(res)


# print(load_txt('../data/train_corpus_seg/Computer/atC19-Computer0001.txt'))

text_t = {}
for i, d in enumerate(dirs):
    # print(i, d)

    if '.DS_Store' in d:
        os.remove(d)
        continue
    files = os.listdir(d)
    files = [os.path.join(d, x) for x in files if x.endswith('txt') and not x.startswith('.')]
    text_t[i] = [load_txt(f) for f in files]

flen = [len(t) for t in text_t.values()]

keys = [k for k in text_t.keys()]

labels = np.repeat(keys, flen)

# flatter nested list
merged = list(itertools.chain.from_iterable(text_t.values()))

df = pd.DataFrame({'label': labels, 'seg_word': merged})

df['seg_word'] = df['seg_word'].str.replace(r'\W+', ' ', flags=re.U)
df['seg_word'] = df['seg_word'].str.replace(r'[A-Za-z]+', ' ENG ')  # 英文轉 ENG

textraw = df.seg_word.values.tolist()


sentences = word2vec.array(textraw)
model = word2vec.Word2Vec(sentences, size=250)

# Save our model.
model.save_word2vec_format("med250.model.bin", binary=True)
# model.save("med250.model.bin")

