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
with open(save_path + 'sr_word2id.pkl', 'rb') as inp:
    sr_id2word = pickle.load(inp)
    sr_word2id = pickle.load(inp)
dict_word2id = dict()
for i in xrange(len(sr_word2id)):
    dict_word2id[sr_word2id.index[i]] = sr_word2id.values[i]


def get_id(word):
    """获取 word 所对应的 id.
    如果该词不在词典中，用 <UNK>（对应的 ID 为 1 ）进行替换。
    """
    if word not in dict_word2id:
        return 1
    else:
        return dict_word2id[word]


def get_id4words(words):
    """把 words 转为 对应的 id"""
    words = words.strip().split(',')  # 先分开词
    ids = map(get_id, words)  # 获取id
    return ids


def test_word2id():
    """把测试集的所有词转成对应的id。"""
    time0 = time.time()
    print('Processing eval data.')
    df_eval = pd.read_csv('../raw_data/question_eval_set.txt', sep='\t', usecols=[0, 2, 4],
                          names=['question_id', 'word_title', 'word_content'], dtype={'question_id': object})
    print('test question number %d' % len(df_eval))
    # 没有 title 的问题用 content 来替换
    na_title_indexs = list()
    for i in xrange(len(df_eval)):
        word_title = df_eval.word_title.values[i]
        if type(word_title) is float:
            na_title_indexs.append(i)
    print('There are %d test questions without title.' % len(na_title_indexs))
    for na_index in na_title_indexs:
        df_eval.at[na_index, 'word_title'] = df_eval.at[na_index, 'word_content']
    # 没有 content 的问题用 title 来替换
    na_content_indexs = list()
    for i in tqdm(xrange(len(df_eval))):
        word_content = df_eval.word_content.values[i]
        if type(word_content) is float:
            na_content_indexs.append(i)
    print('There are %d test questions without content.' % len(na_content_indexs))
    for na_index in tqdm(na_content_indexs):
        df_eval.at[na_index, 'word_content'] = df_eval.at[na_index, 'word_title']
    # 转为 id 形式
    p = Pool()
    eval_title = np.asarray(p.map(get_id4words, df_eval.word_title.values))
    np.save('../data/wd_eval_title.npy', eval_title)
    eval_content = np.asarray(p.map(get_id4words, df_eval.word_content.values))
    np.save('../data/wd_eval_content.npy', eval_content)
    p.close()
    p.join()
    print('Finished changing the eval words to ids. Costed time %g s' % (time.time() - time0))


def train_word2id():
    """把训练集的所有词转成对应的id。"""
    time0 = time.time()
    print('Processing train data.')
    df_train = pd.read_csv('../raw_data/question_train_set.txt', sep='\t', usecols=[0, 2, 4],
                           names=['question_id', 'word_title', 'word_content'], dtype={'question_id': object})
    print('training question number %d ' % len(df_train))
    # 没有 content 的问题用 title 来替换
    na_content_indexs = list()
    for i in tqdm(xrange(len(df_train))):
        word_content = df_train.word_content.values[i]
        if type(word_content) is float:
            na_content_indexs.append(i)
    print('There are %d train questions without content.' % len(na_content_indexs))
    for na_index in tqdm(na_content_indexs):
        df_train.at[na_index, 'word_content'] = df_train.at[na_index, 'word_title']
    # 没有 title 的问题， 丢弃
    na_title_indexs = list()
    for i in xrange(len(df_train)):
        word_title = df_train.word_title.values[i]
        if type(word_title) is float:
            na_title_indexs.append(i)
    print('There are %d train questions without title.' % len(na_title_indexs))
    df_train = df_train.drop(na_title_indexs)
    print('After dropping, training question number(should be 2999952) = %d' % len(df_train))
    # 转为 id 形式
    p = Pool()
    train_title = np.asarray(p.map(get_id4words, df_train.word_title.values))
    np.save('../data/wd_train_title.npy', train_title)
    train_content = np.asarray(p.map(get_id4words, df_train.word_content.values))
    np.save('../data/wd_train_content.npy', train_content)
    p.close()
    p.join()
    print('Finished changing the training words to ids. Costed time %g s' % (time.time() - time0))


if __name__ == '__main__':
    test_word2id()
    train_word2id()
