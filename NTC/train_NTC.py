#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@desc:
@author: TsungHan Yu
@contact: nick.yu@hzn.com.tw
@software: PyCharm  @since:python 3.6.0 on 2017/7/23
"""
import json

import jieba
import re
import requests
from keras.models import model_from_json
from os.path import expanduser
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import gensim
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping


def sent_vec(text):
    # load stopwords set
    stopwordset = set()
    with open('jieba_dict/stopwords.txt', 'r', encoding='utf-8') as sw:
        for line in sw:
            stopwordset.add(line.strip('\n'))

    words = jieba.cut(text, cut_all=False)
    word_list = ''
    for word in words:
        if word not in stopwordset:
            word_list += word + ' '
    return word_list

jsonData = requests.get('http://52.187.15.181/api/?limit=600').text
text = json.loads(jsonData)
arr = text['data']
variables = arr[0].keys()
df = pd.DataFrame([[i[j] for j in variables] for i in arr], columns=variables)
df = pd.DataFrame({'category': df.category, 'text': df.title + df.content})
# nb_classes = len(df.category.unique())

# 非正常字符轉空格
df['text'] = df['text'].str.replace(u'\W+', ' ', flags=re.U)
# 英文轉ENG
df['text'] = df['text'].str.replace(r'[A-Za-z]+', ' ENG ')
# 數字轉NUM
df['text'] = df['text'].str.replace(r'\d+', ' NUM ')

df.text = df.text.map(sent_vec)

df.to_csv('data.csv', encoding='utf8')

def to_one_hot(x):
    for i, d in enumerate(df.category.unique()):
        if d == x:
            return i


# 詞向量空間維度
EMBEDDING_DIM = 200
# 每條文本最大長度
MAX_SEQUENCE_LENGTH = 200
# word2vec模型
# VECTOR_DIR = 'med250.model.bin'
VECTOR_DIR = expanduser('~') + '/model200/med250.model.bin'

# df = pd.read_csv('data.csv')

cateDic = {}
for i, d in enumerate(df.category.unique()):
    cateDic[i] = d

textraw = df.text.values.tolist()

# keras處理token
maxfeatures = 5000  # 只選擇重要的詞

token = Tokenizer(num_words=maxfeatures)
token.fit_on_texts(textraw)  # 如果文本較大可以使用文本流
text_seq = token.texts_to_sequences(textraw)
word_index = token.word_index
print(df.category)
df.category = df.category.map(to_one_hot)
y = df.category.values  # 定義好標籤
print(y)
nb_classes = len(df.category.unique())

print(nb_classes)
print('@@@@@@@@@@@@@@@@')

train_X, test_X, train_y, test_y = train_test_split(text_seq, y, train_size=0.9, random_state=1)

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
try:
    model = model_from_json(open('ASaveModel/my_model_architecture.json').read())
    print("載入模型成功!繼續訓練模型")
except:
    print("載入模型失敗!開始訓練一個新模型")
    model = Sequential()

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    model.add(embedding_layer)
    # model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
    # model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
    # model.add(MaxPooling1D(3))
    # model.add(Dropout(0.3))
    # # model.add(Conv1D(256, 2, padding='valid', activation='relu', strides=1))
    # # model.add(MaxPooling1D(3))
    # model.add(Flatten())
    model.add(LSTM(32))
    # model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    json_string = model.to_json()
    open('SaveModel/my_model_architecture.json', 'w').write(json_string)

model.summary()
try:
    model.load_weights("SaveModel/model_weights.h5")
    print("載入模型成功!繼續訓練模型")
except:
    print("載入模型失敗!開始訓練一個新模型")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
model.fit(X_train, Y_train, validation_split=0.1, epochs=100, batch_size=8, verbose=2, callbacks=[early_stopping])  # , callbacks=[early_stopping]

model.save_weights('SaveModel/model_weights.h5')
print(model.evaluate(X_test, Y_test))
