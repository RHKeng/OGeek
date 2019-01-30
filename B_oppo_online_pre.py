#!/usr/bin/env python
# -*-coding:utf-8-*-

'''

'''

import numpy as np
import pandas as pd
import time
import datetime
import gc
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
import math
from sklearn.metrics import f1_score
import jieba
import jieba.posseg as psg
from collections import Counter
import functools
from gensim.models import word2vec
import Levenshtein

def get_float_list(x):
    return_list = []
    for temp in x:
        return_list.append(float(temp))
    return return_list

# 处理跟query_prediction相关的统计特征
def get_query_prediction_feature(df):
    df['query_prediction_dict'] = df['query_prediction'].map(lambda x : eval(x))
    df['query_prediction_keys'] = df['query_prediction_dict'].map(lambda x : list(x.keys()))
    df['query_prediction_values'] = df['query_prediction_dict'].map(lambda x : get_float_list(list(x.values())))
    df['query_prediction_number'] = df['query_prediction_keys'].map(lambda x : len(x))
    df['query_prediction_max'] = df['query_prediction_values'].map(lambda x : np.nan if len(x) == 0 else np.max(x))
    df['query_prediction_min'] = df['query_prediction_values'].map(lambda x : np.nan if len(x) == 0 else np.min(x))
    df['query_prediction_mean'] = df['query_prediction_values'].map(lambda x : np.nan if len(x) == 0 else np.mean(x))
    df['query_prediction_std'] = df['query_prediction_values'].map(lambda x : np.nan if len(x) == 0 else np.std(x))
    return df

def getBayesSmoothParam(origion_rate):
    origion_rate_mean = origion_rate.mean()
    origion_rate_var = origion_rate.var()
    alpha = origion_rate_mean / origion_rate_var * (origion_rate_mean * (1 - origion_rate_mean) - origion_rate_var)
    beta = (1 - origion_rate_mean) / origion_rate_var * (origion_rate_mean * (1 - origion_rate_mean) - origion_rate_var)
#     print('origion_rate_mean : ', origion_rate_mean)
#     print('origion_rate_var : ', origion_rate_var)
#     print('alpha : ', alpha)
#     print('beta : ', beta)
    return alpha, beta

# 统计单维度的转化率特征
def get_single_dimension_rate_feature(train_df, fea_set):
    for fea in fea_set:
        train_temp_df = pd.DataFrame()
        for index, (train_index, test_index) in enumerate(skf.split(train_df, train_df['label'])):
            temp_df = train_df[[fea, 'label']].iloc[train_index].copy()
            temp_pivot_table = pd.pivot_table(temp_df, index=fea, values='label', aggfunc={len, np.mean, np.sum})
            temp_pivot_table.reset_index(inplace=True)
            temp_pivot_table.rename(columns={'len':fea + '_count', 'mean':fea + '_rate', 'sum':fea + '_click_number'}, inplace=True)
            alpha, beta = getBayesSmoothParam(temp_pivot_table[fea + '_rate'])
            temp_pivot_table[fea + '_rate'] = (temp_pivot_table[fea + '_click_number'] + alpha) / (temp_pivot_table[fea + '_count'] + alpha + beta)
#             del temp_pivot_table[fea + '_click_number']
            fea_df = train_df.iloc[test_index].copy()
            fea_df = pd.merge(fea_df, temp_pivot_table, on=fea, how='left')
#             print(fea_df.head())
            train_temp_df = pd.concat([train_temp_df, fea_df])
#         temp_df = train_df[[fea, 'label']].copy()
#         temp_pivot_table = pd.pivot_table(temp_df, index=fea, values='label', aggfunc={len, np.mean, np.sum})
#         temp_pivot_table.reset_index(inplace=True)
#         temp_pivot_table.rename(columns={'len':fea + '_count', 'mean':fea + '_rate', 'sum':fea + '_click_number'}, inplace=True)
#         alpha, beta = getBayesSmoothParam(temp_pivot_table[fea + '_rate'])
#         temp_pivot_table[fea + '_rate'] = (temp_pivot_table[fea + '_click_number'] + alpha) / (temp_pivot_table[fea + '_count'] + alpha + beta)
# #             del temp_pivot_table[fea + '_click_number']
#         valid_df = pd.merge(valid_df, temp_pivot_table, on=fea, how='left')
        print(fea + ' : finish!!!')
        train_df = train_temp_df
        train_df.sort_index(by='index', ascending=True, inplace=True)
    return train_df

# 统计双维度交叉转化率
def get_jiaocha_dimension_rate_feature(train_df, fea_set):
    for i in range(len(fea_set)):
        for j in range((i+1), len(fea_set)):
            fea1 = fea_set[i]
            fea2 = fea_set[j]
            train_temp_df = pd.DataFrame()
            for index, (train_index, test_index) in enumerate(skf.split(train_df, train_df['label'])):
                temp_df = train_df[[fea1, fea2, 'label']].iloc[train_index].copy()
                temp_pivot_table = pd.pivot_table(temp_df, index=[fea1, fea2], values='label', aggfunc={len, np.mean, np.sum})
                temp_pivot_table.reset_index(inplace=True)
                temp_pivot_table.rename(columns={'len':fea1 + '_' + fea2 + '_count', 'mean':fea1 + '_' + fea2 + '_rate', 'sum':fea1 + '_' + fea2 + '_click_number'}, inplace=True)
                alpha, beta = getBayesSmoothParam(temp_pivot_table[fea1 + '_' + fea2 + '_rate'])
                temp_pivot_table[fea1 + '_' + fea2 + '_rate'] = (temp_pivot_table[fea1 + '_' + fea2 + '_click_number'] + alpha) / (temp_pivot_table[fea1 + '_' + fea2 + '_count'] + alpha + beta)
#                 del temp_pivot_table[fea1 + '_' + fea2 + '_click_number']
                fea_df = train_df.iloc[test_index].copy()
                fea_df = pd.merge(fea_df, temp_pivot_table, on=[fea1, fea2], how='left')
                train_temp_df = pd.concat([train_temp_df, fea_df])
#             temp_df = train_df[[fea1, fea2, 'label']].copy()
#             temp_pivot_table = pd.pivot_table(temp_df, index=[fea1, fea2], values='label', aggfunc={len, np.mean, np.sum})
#             temp_pivot_table.reset_index(inplace=True)
#             temp_pivot_table.rename(columns={'len':fea1 + '_' + fea2 + '_count', 'mean':fea1 + '_' + fea2 + '_rate', 'sum':fea1 + '_' + fea2 + '_click_number'}, inplace=True)
#             alpha, beta = getBayesSmoothParam(temp_pivot_table[fea1 + '_' + fea2 + '_rate'])
#             temp_pivot_table[fea1 + '_' + fea2 + '_rate'] = (temp_pivot_table[fea1 + '_' + fea2 + '_click_number'] + alpha) / (temp_pivot_table[fea1 + '_' + fea2 + '_count'] + alpha + beta)
# #             del temp_pivot_table[fea1 + '_' + fea2 + '_click_number']
            print(fea1 + '_' + fea2 + ' : finish!!!')
#             valid_df = pd.merge(valid_df, temp_pivot_table, on=[fea1, fea2], how='left')
            train_df = train_temp_df
            train_df.sort_index(by='index', ascending=True, inplace=True)
    return train_df

# 统计一些是否交叉的特征
def get_is_title_in_query_feature(df):
    x = df['title']
    y = df['query_prediction_keys']
    is_title_in_query = np.nan
    if len(y) > 0:
        if x in y:
            is_title_in_query = 1
        else:
            is_title_in_query = 0
    return is_title_in_query

def get_is_prefix_in_title_feature(df):
    x = df['prefix']
    y = df['title']
    is_prefix_in_title = np.nan
    if x in y:
        is_prefix_in_title = 1
    else:
        is_prefix_in_title = 0
    return is_prefix_in_title

def get_key_len_list(x):
    return_list = []
    for temp in x:
        return_list.append(len(temp))
    return return_list

# 统计一些跟字符串长度相关的特征
def get_string_len_feature(df):
    df['prefix_len'] = df['prefix'].map(lambda x : len(x))
    df['title_len'] = df['title'].map(lambda x : len(x))
    df['query_prediction_key_len_list'] = df['query_prediction_keys'].map(lambda x : get_key_len_list(x))
    df['query_prediction_key_len_max'] = df['query_prediction_key_len_list'].map(lambda x : np.nan if len(x) == 0 else np.max(x))
    df['query_prediction_key_len_min'] = df['query_prediction_key_len_list'].map(lambda x : np.nan if len(x) == 0 else np.min(x))
    df['query_prediction_key_len_mean'] = df['query_prediction_key_len_list'].map(lambda x : np.nan if len(x) == 0 else np.mean(x))
    df['query_prediction_key_len_std'] = df['query_prediction_key_len_list'].map(lambda x : np.nan if len(x) == 0 else np.std(x))
    df['len_title-prefix'] = df['title_len'] - df['prefix_len']
    df['len_prefix/title'] = df['prefix_len'] / df['title_len']
    df['len_mean-title'] = df['query_prediction_key_len_mean'] - df['title_len']
    df['len_mean/title'] = df['query_prediction_key_len_mean'] / df['title_len']
    del df['query_prediction_key_len_list']
    return df

# 统计title跟prefix的编辑距离
def get_title_prefix_levenshtein_distance(df):
    title = df['title']
    prefix = df['prefix']
    return Levenshtein.distance(title, prefix)

def get_title_prefix_levenshtein_distance_rate(df):
    title_prefix_leven = df['title_prefix_leven']
    title = df['title']
    return (title_prefix_leven / (len(title) + 3))

# 统计title跟query_prediction编辑距离相关的特征
def get_title_query_levenshtein_distance_list(df):
    query_keys_list = df['query_prediction_keys']
    query_values_list = df['query_prediction_values']
    title = df['title']
    return_list = list()
    for i in range(len(query_keys_list)):
        distance = Levenshtein.distance(title, query_keys_list[i])
        return_list.append(distance * query_values_list[i])
    return return_list

def get_title_query_levenshtein_distance_feature(df):
    df['title_query_leven_list'] = df[['query_prediction_keys', 'query_prediction_values', 'title']].apply(get_title_query_levenshtein_distance_list, axis=1)
    df['title_query_leven_sum'] = df['title_query_leven_list'].map(lambda x : np.nan if len(x) == 0 else np.sum(x))
    df['title_query_leven_max'] = df['title_query_leven_list'].map(lambda x : np.nan if len(x) == 0 else np.max(x))
    df['title_query_leven_min'] = df['title_query_leven_list'].map(lambda x : np.nan if len(x) == 0 else np.min(x))
    df['title_query_leven_mean'] = df['title_query_leven_list'].map(lambda x : np.nan if len(x) == 0 else np.mean(x))
    df['title_query_leven_std'] = df['title_query_leven_list'].map(lambda x : np.nan if len(x) == 0 else np.std(x))
    return df

#分词方法，调用结巴接口
def jieba_seg_to_list(sentence, pos=False):
    if not pos:
        #不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        #进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list

#去除干扰词
def jieba_word_filter(seg_list, pos=False):

    filter_list = []
    #根据pos参数选择是否词性过滤
    #不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        filter_list.append(word)
    return filter_list

def jieba_word_deal(sentence, pos=False):
    #调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    seg_list = jieba_seg_to_list(sentence, pos)
    filter_list = jieba_word_filter(seg_list, pos)
    return filter_list

def get_prefix_prediction_key_sentences(x):
    prefix_prediction_key_sentences = ""
    for temp in x:
        if len(prefix_prediction_key_sentences) > 0:
            prefix_prediction_key_sentences = prefix_prediction_key_sentences + temp
        else:
            prefix_prediction_key_sentences = temp
    return prefix_prediction_key_sentences

def get_max_query_key_sentences(x):
    if len(x) == 0:
        return ""
    else:
        return max(x, key=x.get)

def get_jieba_word(df):
    df['query_prediction_key_sentences'] = df['query_prediction_keys'].map(lambda x : get_prefix_prediction_key_sentences(x))
#     df['query_prediction_key_sentences'] = df['query_prediction_dict'].map(lambda x : get_max_query_key_sentences(x))
    df['query_prediction_key_jieba_words'] = df['query_prediction_key_sentences'].map(lambda x : jieba_word_deal(x, False))
    df['query_prediction_words'] = df['query_prediction_keys'].map(lambda x : [jieba_word_deal(j, False) for j in x] if len(x) > 0 else np.nan)
    df['title_jieba_words'] = df['title'].map(lambda x : jieba_word_deal(x, False))
    df['prefix_jieba_words'] = df['prefix'].map(lambda x : jieba_word_deal(x, False))
#     del df['query_prediction_key_sentences']
    return df

def word_match_share(df):
    q1words = {}
    q2words = {}
    for word in df[0]:
        q1words[word] = 1
    for word in df[1]:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def jaccard(df):
    wic = set(df[0]).intersection(set(df[1]))
    uw = set(df[0]).union(df[1])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

def common_words(df):
    return len(set(df[0]).intersection(set(df[1])))

def total_unique_words(df):
    return len(set(df[0]).union(df[1]))

def wc_diff(df):
    return abs(len(df[0]) - len(df[1]))

def wc_ratio(df):
    l1 = len(df[0])*1.0
    l2 = len(df[1])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(df):
    return abs(len(set(df[0])) - len(set(df[1])))

def wc_ratio_unique(df):
    l1 = len(set(df[0])) * 1.0
    l2 = len(set(df[1]))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def tfidf_word_match_share(df, weights=None):
    q1words = {}
    q2words = {}
    for word in df[0]:
        q1words[word] = 1
    for word in df[1]:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def deal_word_for_all(train_df, fea1, fea2, func, colName):
    train_df[colName] = train_df[[fea1, fea2]].apply(func, axis=1)
#     valid_df[colName] = valid_df[[fea1, fea2]].apply(func, axis=1)
    print(colName + ' finish!!!')
    return train_df

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

def get_word_statistic_feature(train_df, col_list):
#     df = pd.concat([train_df[['query_prediction_key_jieba_words', 'title_jieba_words', 'prefix_jieba_words']], valid_df[['query_prediction_key_jieba_words', 'title_jieba_words', 'prefix_jieba_words']]])
#     train_qs = pd.Series(df['query_prediction_key_jieba_words'].tolist() + df['title_jieba_words'].tolist() + df['prefix_jieba_words'].tolist())
#     words = [x for y in train_qs for x in y]
#     counts = Counter(words)
#     weights = {word: get_weight(count) for word, count in counts.items()}
    for col in col_list:
        fea1 = col[0]
        fea2 = col[1]
        train_df = deal_word_for_all(train_df, fea1, fea2, word_match_share, fea1[0] + '_' + fea2[0] + '_word_match')
        train_df = deal_word_for_all(train_df, fea1, fea2, jaccard, fea1[0] + '_' + fea2[0] + '_jaccard')
        train_df = deal_word_for_all(train_df, fea1, fea2, common_words, fea1[0] + '_' + fea2[0] + '_common_words')
        train_df = deal_word_for_all(train_df, fea1, fea2, total_unique_words, fea1[0] + '_' + fea2[0] + '_total_unique_words')
        train_df = deal_word_for_all(train_df, fea1, fea2, wc_diff, fea1[0] + '_' + fea2[0] + '_wc_diff')
        train_df = deal_word_for_all(train_df, fea1, fea2, wc_ratio, fea1[0] + '_' + fea2[0] + '_wc_ratio')
        train_df = deal_word_for_all(train_df, fea1, fea2, wc_diff_unique, fea1[0] + '_' + fea2[0] + '_wc_diff_unique')
        train_df = deal_word_for_all(train_df, fea1, fea2, wc_ratio_unique, fea1[0] + '_' + fea2[0] + '_wc_ratio_unique')
#         f = functools.partial(tfidf_word_match_share, weights=weights)
#         train_df, valid_df = deal_word_for_all(train_df, valid_df, fea1, fea2, f, fea1[0] + '_' + fea2[0] + '_tfidf_word_match_share')
    return train_df

def get_w2v_array(word_list, word_wv, num_features):
    word_vectors = np.zeros((len(word_list), num_features))
    for i in range(len(word_list)):
        word_vectors[i][:] = word_wv[str(word_list[i])]
    mean_array = np.mean(word_vectors, axis=0)
    return mean_array

def get_title_prefix_similarity(df, f_similarity):
    title_array = df['title_jieba_array']
    prefix_array = df['prefix_jieba_array']
    similarity = 0
    if f_similarity == 'dot':
        similarity = np.dot(title_array, prefix_array)
    elif f_similarity == 'norm':
        similarity = np.linalg.norm(title_array - prefix_array)
    else:
        similarity = np.dot(title_array,prefix_array) / (np.linalg.norm(title_array) * np.linalg.norm(prefix_array))
    return similarity

def get_title_query_similarity_list(df, f_similarity, word_wv, num_features):
    title_array = df['title_jieba_array']
    query_prediction_words = df['query_prediction_words']
    query_prediction_keys = df['query_prediction_keys']
    query_prediction_dict = df['query_prediction_dict']
    similarity_list = list()
    if len(query_prediction_keys) <= 0:
        return similarity_list
    if f_similarity == 'dot':
        i = 0
        for key in query_prediction_keys:
            key_array = get_w2v_array(query_prediction_words[i], word_wv, num_features)
            similarity = np.dot(title_array, key_array) * float(query_prediction_dict[key])
            similarity_list.append(similarity)
            i = i + 1
    elif f_similarity == 'norm':
        i = 0
        for key in query_prediction_keys:
            key_array = get_w2v_array(query_prediction_words[i], word_wv, num_features)
            similarity = np.linalg.norm(title_array - key_array) * float(query_prediction_dict[key])
            similarity_list.append(similarity)
            i = i + 1
    else:
        i = 0
        for key in query_prediction_keys:
            key_array = get_w2v_array(query_prediction_words[i], word_wv, num_features)
            similarity = (np.dot(title_array, key_array) / (np.linalg.norm(title_array) * np.linalg.norm(key_array))) * float(query_prediction_dict[key])
            similarity_list.append(similarity)
            i = i + 1
    return similarity_list

def get_similarity_feature(train_df):
    f_list = ['dot', 'norm', 'cosine']
    for fun in f_list:
        f_prefix_similarity = functools.partial(get_title_prefix_similarity, f_similarity=fun)
        train_df['title_prefix_' + fun + '_similarity'] = train_df[['title_jieba_array', 'prefix_jieba_array']].apply(f_prefix_similarity, axis=1)
#         f_query_similarity = functools.partial(get_title_query_similarity, f_similarity=fun, word_wv=word_wv, num_features=num_features)
#         train_df['title_query_' + fun + '_similarity'] = train_df[['title_jieba_array', 'query_prediction_words', 'query_prediction_keys', 'query_prediction_dict']].apply(f_query_similarity, axis=1)
#         valid_df['title_query_' + fun + '_similarity'] = valid_df[['title_jieba_array', 'query_prediction_words', 'query_prediction_keys', 'query_prediction_dict']].apply(f_query_similarity, axis=1)
        f_query_similarity_list = functools.partial(get_title_query_similarity_list, f_similarity=fun, word_wv=word_wv, num_features=num_features)
        train_df['title_query_' + fun + '_similarity_list'] = train_df[['title_jieba_array', 'query_prediction_words', 'query_prediction_keys', 'query_prediction_dict']].apply(f_query_similarity_list, axis=1)
        train_df['title_query_' + fun + '_similarity'] = train_df['title_query_' + fun + '_similarity_list'].map(lambda x : np.nan if len(x)==0 else np.sum(x))
        train_df['title_query_' + fun + '_similarity_max'] = train_df['title_query_' + fun + '_similarity_list'].map(lambda x : np.nan if len(x)==0 else np.max(x))
        train_df['title_query_' + fun + '_similarity_min'] = train_df['title_query_' + fun + '_similarity_list'].map(lambda x : np.nan if len(x)==0 else np.min(x))
        train_df['title_query_' + fun + '_similarity_mean'] = train_df['title_query_' + fun + '_similarity_list'].map(lambda x : np.nan if len(x)==0 else np.mean(x))
        train_df['title_query_' + fun + '_similarity_std'] = train_df['title_query_' + fun + '_similarity_list'].map(lambda x : np.nan if len(x)==0 else np.std(x))
        print(fun + ' : finish!!!')
    return train_df

# 导出特征工程文件
def exportDf(df, fileName):
    df.to_csv('../temp/%s.csv' % fileName, header=True, index=True)

def main():

    # 开始导入数据
    print("~~~~~~~~~~~~~~~~~~~~~~开始导入数据~~~~~~~~~~~~~~~~~~~~~~~~~~")
    train_df = pd.read_table('../data/oppo_round1_train_20180929.txt', names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, quoting=3)
    valid_df = pd.read_table('../data/oppo_round1_vali_20180929.txt', names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, quoting=3)
    train_df = pd.concat([train_df, valid_df])
    train_df.reset_index(inplace=True)
    train_df['index'] = train_df.index

    print("~~~~~~~~~~~~~~~~~~~~~~统计特征~~~~~~~~~~~~~~~~~~~~~~~~~~")
    train_df = get_query_prediction_feature(train_df)
    skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
    fea_set = ['prefix', 'title', 'tag', 'query_prediction']
    train_df = get_single_dimension_rate_feature(train_df, fea_set)
    jiaocha_fea_set = ['prefix', 'title', 'tag']
    train_df = get_jiaocha_dimension_rate_feature(train_df, jiaocha_fea_set)
    train_df['is_title_in_query'] = train_df[['title', 'query_prediction_keys']].apply(get_is_title_in_query_feature, axis = 1)
    train_df['is_prefix_in_title'] = train_df[['prefix', 'title']].apply(get_is_prefix_in_title_feature, axis = 1)
    train_df = get_string_len_feature(train_df)

    print("~~~~~~~~~~~~~~~~~~~~~~编辑距离特征~~~~~~~~~~~~~~~~~~~~~~~~~~")
    train_df['title_prefix_leven'] = train_df[['title', 'prefix']].apply(get_title_prefix_levenshtein_distance, axis=1)
    train_df['title_prefix_leven_rate'] = train_df[['title', 'title_prefix_leven']].apply(get_title_prefix_levenshtein_distance_rate, axis=1)
    train_df = get_title_query_levenshtein_distance_feature(train_df)

    print("~~~~~~~~~~~~~~~~~~~~~~分词~~~~~~~~~~~~~~~~~~~~~~~~~~")
    train_df = get_jieba_word(train_df)

    print("~~~~~~~~~~~~~~~~~~~~~~距离特征~~~~~~~~~~~~~~~~~~~~~~~~~~")
    col_list = [['query_prediction_key_jieba_words', 'title_jieba_words'], ['prefix_jieba_words', 'title_jieba_words'], ['prefix_jieba_words', 'query_prediction_key_jieba_words']]
    train_df = get_word_statistic_feature(train_df, col_list)

    print("~~~~~~~~~~~~~~~~~~~~~~w2v相关特征~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Set values for various parameters
    num_features = 500  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 20       # Number of threads to run in parallel
    context = 5          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    word2vec_df = train_df[['query_prediction_words', 'title_jieba_words', 'prefix_jieba_words', 'query_prediction_number']]
    word2vec_df.reset_index(inplace=True)
    word2vec_list = word2vec_df['title_jieba_words'].tolist() + word2vec_df['prefix_jieba_words'].tolist() + [y for x in word2vec_df['query_prediction_words'][word2vec_df.query_prediction_number > 0] for y in x]
    model = word2vec.Word2Vec(word2vec_list, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    word_wv = model.wv
    train_df['title_jieba_array'] = train_df['title_jieba_words'].map(lambda x : get_w2v_array(x, word_wv, num_features))
    train_df['prefix_jieba_array'] = train_df['prefix_jieba_words'].map(lambda x : get_w2v_array(x, word_wv, num_features))
    train_df = get_similarity_feature(train_df)
    # 保存训练的词向量模型
    model.save("../temp/B_word2vec.model")
    fea = [
        'query_prediction_number', 'query_prediction_max', 'query_prediction_min', 'query_prediction_mean', 'query_prediction_std',
           'prefix_count', 'prefix_rate',
     'title_count', 'title_rate', 'tag_count', 'tag_rate',
     'query_prediction_count', 'query_prediction_rate', 'prefix_title_count',
     'prefix_title_rate',  'prefix_tag_count', 'prefix_tag_rate',
     'title_tag_count', 'title_tag_rate',
        'prefix_click_number', 'title_click_number', 'query_prediction_click_number', 'prefix_tag_click_number',
        'prefix_title_click_number', 'title_tag_click_number',
        'is_title_in_query', 'is_prefix_in_title',
    #     'title_tag_types', 'prefix_tag_types', 'tag_title_types', 'tag_prefix_types',
    #  'title_prefix_types', 'prefix_title_types', 'tag_query_prediction_types', 'title_query_prediction_types',
          'prefix_len', 'title_len',
     'query_prediction_key_len_max', 'query_prediction_key_len_min',
     'query_prediction_key_len_mean', 'query_prediction_key_len_std',
     'len_title-prefix', 'len_prefix/title', 'len_mean-title', 'len_mean/title',
    #     'q_t_jaccard', 'p_t_jaccard', 'p_q_jaccard',
        'q_t_word_match', 'q_t_common_words',
    #     'q_t_tfidf_word_match_share', 'p_t_tfidf_word_match_share', 'p_q_tfidf_word_match_share',
     'q_t_total_unique_words', 'q_t_wc_diff', 'q_t_wc_ratio',
     'q_t_wc_diff_unique', 'q_t_wc_ratio_unique',
     'p_t_word_match', 'p_t_common_words',
     'p_t_total_unique_words', 'p_t_wc_diff', 'p_t_wc_ratio',
     'p_t_wc_diff_unique', 'p_t_wc_ratio_unique',
     'p_q_word_match', 'p_q_common_words',
     'p_q_total_unique_words', 'p_q_wc_diff', 'p_q_wc_ratio',
     'p_q_wc_diff_unique', 'p_q_wc_ratio_unique',
        'title_prefix_dot_similarity',
     'title_query_dot_similarity', 'title_prefix_norm_similarity',
     'title_query_norm_similarity', 'title_prefix_cosine_similarity',
     'title_query_cosine_similarity',
        'title_query_dot_similarity_max', 'title_query_dot_similarity_min',
     'title_query_dot_similarity_mean', 'title_query_dot_similarity_std',
        'title_query_norm_similarity_min', 'title_query_norm_similarity_mean',
     'title_query_norm_similarity_std',
        'title_query_cosine_similarity_max', 'title_query_cosine_similarity_min',
     'title_query_cosine_similarity_mean', 'title_query_cosine_similarity_std',
        'title_prefix_leven', 'title_prefix_leven_rate',
     'title_query_leven_sum', 'title_query_leven_max', 'title_query_leven_min',
     'title_query_leven_mean', 'title_query_leven_std',
        'prefix', 'query_prediction', 'title', 'tag', 'index', 'label'
          ]

    exportDf(train_df[fea], 'A_final_train_online_df')

    print("~~~~~~~~~~~~~~~~~~~~~~finish!!!!~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == '__main__':
    main()
