#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
@desc:  
@author: TsungHan Yu  
@contact: nick.yu@hzn.com.tw  
@software: PyCharm  @since:python 3.6.0 on 2017/7/3
"""

import os
import jieba


# 將內容保存到 `save_path`
def save_file(save_path, content):
    with open(save_path, "w", encoding='utf8') as fp:
        fp.write(content)


# 讀取文件
def read_file(path):
    with open(path, "r", encoding='utf8') as fp:
        content = fp.read()
    return content


# 將文章做分詞
# article_path  -> 未分詞文章路徑
# seg_path      -> 已分詞文章路徑
def article_segment(article_path, seg_path):
    # os.listdir() 方法返回指定的文件夾包含的文件或文件夾的名字的列表 ['.DS_Store', 'Art', 'Education', 'Literature']
    catelist = os.listdir(article_path)

    # for-in 有哪些目錄
    for mydir in catelist:

        # 目錄路徑      EX: train_corpus/art/
        class_path = article_path + mydir + "/"
        # 分詞後目錄路徑 EX: train_corpus_seg/art/
        seg_dir = seg_path + mydir + "/"

        if mydir == '.DS_Store':
            os.remove(article_path + mydir)

        # 有時候mac會產生 `.DS_Store`
        if mydir[0] == '.':
            print('#' + mydir)
            continue

        # 如果分詞文件夾不存在創建一個
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        file_list = os.listdir(class_path)

        # for-in 目錄下的文件
        for file_path in file_list:

            if file_path == '.DS_Store':
                os.remove(class_path + file_path)

            if file_path[0] == '.':
                print('##', file_path)
                continue

            # 文章路徑  EX: train_corpus/art/21.txt
            fullname = class_path + file_path
            # 文章內容
            content = read_file(fullname)
            # 刪除換行
            content = content.replace('\n', '')
            content = content.replace('\r\n', '')
            # 刪除空格、空行
            content = content.replace(' ', '')
            # 使用 jieba 做分詞
            content_seg = jieba.cut(content)
            # 將分詞完的內容保存
            save_file(seg_dir + file_path, " ".join(content_seg))

    print("完成分詞")


if __name__ == "__main__":
    # 對訓練集分詞

    # 未分詞分類語料庫路徑
    article_path = "../data/train_corpus/"
    # 分詞後分類語料庫路徑
    seg_path = "../data/train_corpus_seg/"
    article_segment(article_path, seg_path)

    # 對測試集分詞

    # 未分詞分類語料庫路徑
    article_path = "../data/test_corpus/"
    # 分詞後分類語料庫路徑
    seg_path = "../data/test_corpus_seg/"
    article_segment(article_path, seg_path)