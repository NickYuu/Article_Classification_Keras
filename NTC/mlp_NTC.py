#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
@desc:  
@author: TsungHan Yu  
@contact: nick.yu@hzn.com.tw  
@software: PyCharm  
@since:python 3.6.0 on 2017/7/26
"""

import pandas as pd
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import EarlyStopping


# keras處理token
max_words = 3000 # 只選擇重要的詞
batch_size = 32
epochs = 5


def to_one_hot(x):
    for i, d in enumerate(df.category.unique()):
        if d == x:
            return i

print('Loading data...')
df = pd.read_csv('data.csv')
cateDic = {}
for i, d in enumerate(df.category.unique()):
    cateDic[i] = d

textraw = df.text.values.tolist()


token = Tokenizer(num_words=max_words)
token.fit_on_texts(textraw)  # 如果文本較大可以使用文本流
text_seq = token.texts_to_sequences(textraw)

df.category = df.category.map(to_one_hot)
y = df.category.values  # 定義好標籤
nb_classes = len(df.category.unique())
tdf = pd.DataFrame({'s': text_seq, 'r': textraw})
train_X, test_X, train_y, test_y = train_test_split(tdf, y, train_size=0.9, random_state=1)

print('Vectorizing sequence data...')
x_train = token.sequences_to_matrix(train_X.s, mode='binary')
x_test = token.sequences_to_matrix(test_X.s, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
# 將y的格式展開為one-hot
Y_train = np_utils.to_categorical(train_y, nb_classes)
Y_test = np_utils.to_categorical(test_y, nb_classes)
print('y_train shape:', Y_train.shape)
print('y_test shape:', Y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
history = model.fit(x_train, Y_train, validation_split=0.2, epochs=50, batch_size=8, verbose=2, callbacks=[early_stopping]) # , callbacks=[early_stopping]

score = model.evaluate(x_test, Y_test, batch_size=batch_size, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

"""
進行預測
"""

prediction = model.predict_classes(x_test)

# print(Y_test.argmax(1))
# print(prediction)
print('=======')
df = pd.DataFrame({'label': Y_test.argmax(1),
                   'predict': prediction})

error_prediction = df[df.label != df.predict]
index = error_prediction.index
print(index)
for i in index:
    print('```')
    print(test_X.reset_index(drop=True).r.loc[i])
    print('```')
    print('### 真實值為: ', cateDic[Y_test.argmax(1)[i]], '預測值為: ', cateDic[prediction[i]])
    print()
    print()
print(cateDic)
matrix = pd.crosstab(df.label, df.predict)
print()
print(matrix)
