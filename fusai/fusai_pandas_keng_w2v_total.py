# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import datetime
import gc
import jieba
import functools
from gensim.models import Word2Vec
import json

#定义jieba分词函数
def jieba_sentences(sentence):
    seg_list = jieba.cut(sentence)
    seg_list = list(seg_list)
    return seg_list

##-----------------------------------------------------------------------------
if __name__=='__main__':
    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    print(now)
    train_df = pd.read_table('../data/data_train.txt', names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, na_values='', keep_default_na=False, encoding='utf-8', quoting=3)
    valid_df = pd.read_table('../data/data_vali.txt', names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, na_values='', keep_default_na=False, encoding='utf-8', quoting=3)
    test_df = pd.read_table('../data/data_test.txt', names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, na_values='', keep_default_na=False, encoding='utf-8', quoting=3)
    total_df = pd.concat([test_df, valid_df, train_df])
    del test_df
    del valid_df
    del train_df
    gc.collect()
    total_df['query_prediction'] = total_df['query_prediction'].map(lambda x : np.nan if x is np.nan else eval(x))
    total_df['query_prediction'] = total_df['query_prediction'].map(lambda x : np.nan if x is np.nan else sorted(x.keys()))
    sentence_list = [t for x in total_df['query_prediction'][total_df.query_prediction.notnull()] for t in x] + total_df['title'].tolist() + total_df['prefix'].tolist()
    sentence_list = [jieba_sentences(x) for x in sentence_list]
    my_model = Word2Vec(sentence_list, size=50, window=5, sg=1, hs=1, min_count=2, workers=1, seed=0)
    my_model.save('../data/keng_940129seed0/w2v_total_final_50wei_1.model')

    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    print(now)
