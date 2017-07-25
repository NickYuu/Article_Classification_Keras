#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
@desc:  
@author: TsungHan Yu  
@contact: nick.yu@hzn.com.tw  
@software: PyCharm  @since:python 3.6.0 on 2017/7/25
"""
import jieba
import re
from gensim import models
import pandas as pd

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


df = pd.read_csv('data.csv')

nb_classes = len(df.category.unique())

# 非正常字符轉空格
df['ttc'] = df['ttc'].str.replace(u'\W+', ' ', flags=re.U)
# 英文轉ENG
df['ttc'] = df['ttc'].str.replace(r'[A-Za-z]+', ' ENG ')
# 數字轉NUM
df['ttc'] = df['ttc'].str.replace(r'\d+', ' NUM ')

df.ttc = df.ttc.map(sent_vec)
df.to_csv('data.csv')
