#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
@desc:  
@author: TsungHan Yu  
@contact: nick.yu@hzn.com.tw  
@software: PyCharm  
@since:python 3.6.0 on 2017/7/25
"""

from os.path import expanduser
import os
import re
import pandas as pd
import numpy as np
import itertools
from keras.preprocessing.text import Tokenizer
import gensim
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn import svm


def to_one_hot(x):
    for i, d in enumerate(df.category.unique()):
        if d == x:
            return i


# 詞向量空間維度
EMBEDDING_DIM = 200
# 每條文本最大長度
MAX_SEQUENCE_LENGTH = 60
# word2vec模型
# VECTOR_DIR = 'med250.model.bin'
VECTOR_DIR = expanduser("~") + '/model200/med250.model.bin'

df = pd.read_csv('data.csv')

cateDic = {}
for i, d in enumerate(df.category.unique()):
    cateDic[i] = d

textraw = df.ttc.values.tolist()

# keras處理token
maxfeatures = 5000  # 只選擇重要的詞

token = Tokenizer(num_words=maxfeatures)
token.fit_on_texts(textraw)  # 如果文本較大可以使用文本流
text_seq = token.texts_to_sequences(textraw)
word_index = token.word_index

df.category = df.category.map(to_one_hot)
y = df.category.values  # 定義好標籤
nb_classes = len(df.category.unique())

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


w2v_model = gensim.models.Word2Vec.load(VECTOR_DIR)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if np.unicode(word) in w2v_model:
        embedding_matrix[i] = np.asarray(w2v_model[np.unicode(word)],
                                         dtype='float32')



clf = svm.SVC(gamma=0.001, C=100.)  # 使用線性核
clf_res = clf.fit(X_train, train_y)
train_pred = clf_res.predict(X_train)
test_pred = clf_res.predict(X_test)

print(len(test_y))
print(len(train_pred[test_pred != test_y]))

# for p, y in zip(train_pred, Y_train):
