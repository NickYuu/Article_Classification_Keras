#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
@desc:  
@author: TsungHan Yu  
@contact: nick.yu@hzn.com.tw  
@software: PyCharm  
@since:python 3.6.0 on 2017/7/25
"""

import pandas as pd

# 清除圖書館

df = pd.read_csv('data.csv')

# print(df[df.category == '影劇、展覽'])
#
# df = df.drop(df[df.category == '影劇、展覽'].index, 0)
# df = df.drop('Unnamed: 0', axis=1)
#
# df.to_csv('data.csv')
# print(df)

print(df.category.value_counts())
