# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import time


save_path = '../data/'
with open(save_path + 'sr_char2id.pkl', 'rb') as inp:
    sr_id2char = pickle.load(inp)
    sr_char2id = pickle.load(inp)
dict_char2id = dict()
for i in xrange(len(sr_char2id)):
    dict_char2id[sr_char2id.index[i]] = sr_char2id.values[i]


def get_id(char):
    """获取 char 所对应的 id.
    如果该字不在字典中，用1进行替换。
    """
    if char not in dict_char2id:
        return 1
    else:
        return dict_char2id[char]


def get_id4chars(chars):
    """把 chars 转为 对应的 id"""
    chars = chars.strip().split(',')  # 先分开字
    ids = map(get_id, chars)          # 获取id
    return ids


def test_char2id():
    """把测试集的所有字转成对应的id。"""
    time0 = time.time()
    print('Processing eval data.')
    df_eval = pd.read_csv('../raw_data/question_eval_set.txt', sep='\t',  usecols=[0, 1, 3],
                          names=['question_id', 'char_title', 'char_content'], dtype={'question_id': object})
    print('test question number %d' % len(df_eval))
    # 没有 title 的问题用 content 来替换
    na_title_indexs = list()
    for i in xrange(len(df_eval)):
        char_title = df_eval.char_title.values[i]
        if type(char_title) is float:
            na_title_indexs.append(i)
    print('There are %d test questions without title.' % len(na_title_indexs))
    for na_index in na_title_indexs:
        df_eval.at[na_index, 'char_title'] = df_eval.at[na_index, 'char_content']
    # 没有 content 的问题用 title 来替换
    na_content_indexs = list()
    for i in tqdm(xrange(len(df_eval))):
        char_content = df_eval.char_content.values[i]
        if type(char_content) is float:
            na_content_indexs.append(i)
    print('There are %d test questions without content.' % len(na_content_indexs))
    for na_index in tqdm(na_content_indexs):
        df_eval.at[na_index, 'char_content'] = df_eval.at[na_index, 'char_title']
    # 转为 id 形式
    p = Pool()
    eval_title = np.asarray(p.map(get_id4chars, df_eval.char_title.values))
    np.save('../data/ch_eval_title.npy', eval_title)
    eval_content = np.asarray(p.map(get_id4chars, df_eval.char_content.values))
    np.save('../data/ch_eval_content.npy', eval_content)
    p.close()
    p.join()
    print('Finished changing the eval chars to ids. Costed time %g s' % (time.time()-time0))


def train_char2id():
    """把训练集的所有字转成对应的id。"""
    time0 = time.time()
    print('Processing train data.')
    df_train = pd.read_csv('../raw_data/question_train_set.txt', sep='\t', usecols=[0, 1, 3],
                           names=['question_id', 'char_title', 'char_content'], dtype={'question_id': object})
    print('training question number %d ' % len(df_train))
    # 没有 content 的问题用 title 来替换
    na_content_indexs = list()
    for i in tqdm(xrange(len(df_train))):
        char_content = df_train.char_content.values[i]
        if type(char_content) is float:
            na_content_indexs.append(i)
    print('There are %d train questions without content.' % len(na_content_indexs))
    for na_index in tqdm(na_content_indexs):
        df_train.at[na_index, 'char_content'] = df_train.at[na_index, 'char_title']
    # 没有 title 的问题， 与词一样丢弃下面样本
    na_title_indexs = [328877, 422123, 633584, 768738, 818616, 876828, 1273673, 1527297,
              1636237, 1682969, 2052477, 2628516, 2657464, 2904162, 2993517]
    for i in xrange(len(df_train)):
        char_title = df_train.char_title.values[i]
        if type(char_title) is float:
            na_title_indexs.append(i)
    print('There are %d train questions without title.' % len(na_title_indexs))
    df_train = df_train.drop(na_title_indexs)
    print('After dropping, training question number(should be 2999952) = %d' % len(df_train))
    # 转为 id 形式
    p = Pool()
    train_title = np.asarray(p.map(get_id4chars, df_train.char_title.values))
    np.save('../data/ch_train_title.npy', train_title)
    train_content = np.asarray(p.map(get_id4chars, df_train.char_content.values))
    np.save('../data/ch_train_content.npy', train_content)
    p.close()
    p.join()
    print('Finished changing the training chars to ids. Costed time %g s' % (time.time() - time0))


if __name__ == '__main__':
    test_char2id()
    train_char2id()






