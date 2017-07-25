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



df = pd.read_csv('data.csv')

textraw = df.ttc.values.tolist()


sentences = word2vec.array(textraw)
model = word2vec.Word2Vec(sentences, size=250)

# Save our model.
# model.wv.save_word2vec_format("med250.model.bin", binary=True)
model.save("model_250.bin")
