#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
@desc:  
@author: TsungHan Yu  
@contact: nick.yu@hzn.com.tw  
@software: PyCharm  @since:python 3.6.0 on 2017/7/3
"""

from os import path
import os
import re
import codecs
import pandas as pd
import numpy as np
import itertools
from keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

rootdir = '../data/train_corpus_seg/'
dirs = os.listdir(rootdir)
dirs = [path.join(rootdir, f) for f in dirs]


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
    files = [path.join(d, x) for x in files if x.endswith('txt') and not x.startswith('.')]
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

# print(textraw)
# keras處理token
maxfeatures = 50000  # 只選擇重要的詞

token = Tokenizer(num_words=maxfeatures)
token.fit_on_texts(textraw)  # 如果文本較大可以使用文本流
text_seq = token.texts_to_sequences(textraw)

y = df.label.values  # 定義好標籤
nb_classes = len(np.unique(y))
# print(nb_classes)

# pd.DataFrame({'labels': y, 'text': text_seq}).to_csv("data.csv", encoding="utf-8")

# # 定義文本最大長度
# maxlen = 600
# # 批次
# batch_size = 32
# # 詞向量維度
# word_dim = 100
# # 卷積核個數
# nb_filter = 200
# # 卷積窗口大小
# filter_length = 10
# # 隱藏層神經元個數
# hidden_dims = 50
# # 迭代次數
# nb_epoch = 10
# # 池化窗口大小
# pool_length = 50
#
# train_X, test_X, train_y, test_y = train_test_split(text_seq, y, train_size=0.8, random_state=1)
#
# # 轉為等長矩陣，長度為maxlen
# print("Pad sequences (samples x time)")
# X_train = sequence.pad_sequences(train_X, maxlen=maxlen, padding='post', truncating='post')
# X_test = sequence.pad_sequences(test_X, maxlen=maxlen, padding='post', truncating='post')
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
#
# # 將y的格式展開為one-hot
# Y_train = np_utils.to_categorical(train_y, nb_classes)
# Y_test = np_utils.to_categorical(test_y, nb_classes)
#
# # CNN 模型
# print('Build model...')
# model = Sequential()
#
# # 词向量嵌入层，输入：词典大小，词向量大小，文本长度
# model.add(Embedding(maxfeatures, word_dim, input_length=maxlen))
# # model.add(Dropout(0.25))
# model.add(LSTM(100))
#
# model.add(Dense(hidden_dims))
# model.add(Dropout(0.25))
# model.add(Activation('relu'))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
#
# train_history = model.fit(X_train,
#                           Y_train,
#                           batch_size=100,
#                           epochs=10,
#                           verbose=2,
#                           validation_split=0.2)
#
# print('Evaluate Model...')
#
# scores = model.evaluate(X_test, Y_test, verbose=1)
# print()
# print(scores[1])
