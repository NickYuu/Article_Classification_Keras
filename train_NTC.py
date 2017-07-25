#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@desc:
@author: TsungHan Yu
@contact: nick.yu@hzn.com.tw
@software: PyCharm  @since:python 3.6.0 on 2017/7/23
"""
from os import path
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
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
import jieba

def cutword_1(x):
    words = jieba.cut(x)
    return ' '.join(words)

# 詞向量空間維度
EMBEDDING_DIM = 200
# 每條文本最大長度
MAX_SEQUENCE_LENGTH = 400
# word2vec模型
VECTOR_DIR = 'med250.model.bin'

df = pd.read_csv('data.csv')

nb_classes = len(df.category.unique())

df['ttc'] = df['ttc'].str.replace(u'\W+', ' ', flags=re.U)  # 非正常字符转空格
df['ttc'] = df['ttc'].str.replace(r'[A-Za-z]+', ' ENG ')   # 英文转ENG
df['ttc'] = df['ttc'].str.replace(r'\d+', ' NUM ')   # 数字转NUM

