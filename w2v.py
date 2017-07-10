#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
@desc:  
@author: TsungHan Yu  
@contact: nick.yu@hzn.com.tw  
@software: PyCharm  @since:python 3.6.0 on 2017/7/3
"""

EMBEDDING_DIM = 250  # 词向量空间维度
MAX_SEQUENCE_LENGTH = 400  # 每条新闻最大长度
VECTOR_DIR = 'med250.model.bin'  # 词向量模型文件

from os import path
import os
import re
import codecs
import pandas as pd
import numpy as np
import itertools
from keras.preprocessing.text import Tokenizer
import gensim
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential

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
word_index = token.word_index

y = df.label.values  # 定義好標籤
nb_classes = len(np.unique(y))

train_X, test_X, train_y, test_y = train_test_split(text_seq, y, train_size=0.8, random_state=1)

# 轉為等長矩陣，長度為maxlen
print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(train_X, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test = sequence.pad_sequences(test_X, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# 將y的格式展開為one-hot
Y_train = np_utils.to_categorical(train_y, nb_classes)
Y_test = np_utils.to_categorical(test_y, nb_classes)

# ----------------------------------------


w2v_model = gensim.models.Word2Vec.load(VECTOR_DIR)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if np.unicode(word) in w2v_model:
        embedding_matrix[i] = np.asarray(w2v_model[np.unicode(word)],
                                         dtype='float32')
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))
model.summary()
# plot_model(model, to_file='model.png',show_shapes=True)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=128)
print(model.evaluate(X_test, Y_test))
