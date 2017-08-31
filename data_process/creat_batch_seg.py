# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
from multiprocessing import Pool
import sys
import os

sys.path.append('../')
from data_helpers import pad_X30
from data_helpers import pad_X52
from data_helpers import wd_pad_cut_docs
from data_helpers import ch_pad_cut_docs
from data_helpers import train_batch
from data_helpers import eval_batch


wd_train_path = '../data/wd-data/seg_train/'
wd_valid_path = '../data/wd-data/seg_valid/'
wd_test_path = '../data/wd-data/seg_test/'
ch_train_path = '../data/ch-data/seg_train/'
ch_valid_path = '../data/ch-data/seg_valid/'
ch_test_path = '../data/ch-data/seg_test/'
paths = [wd_train_path, wd_valid_path, wd_test_path,
         ch_train_path, ch_valid_path, ch_test_path]
for each in paths:
    if not os.path.exists(each):
        os.makedirs(each)


# word 数据打包
def wd_train_get_batch(title_len=30, batch_size=128):
    print('loading word train_title and train_content, this should cost minutes, please wait.')
    train_title = np.load('../data/wd_train_title.npy')
    train_content = np.load('../data/wd_train_content.npy')
    p = Pool(8)
    X_title = np.asarray(p.map(pad_X30, train_title))
    X_content = np.asarray(p.map(wd_pad_cut_docs, train_content))
    p.close()
    p.join()
    X_content.shape = [-1, 30*10]
    X = np.hstack([X_title, X_content])
    y = np.load('../data/y_tr.npy')
    # 划分验证集
    sample_num = X.shape[0]
    np.random.seed(13)
    valid_num = 100000
    new_index = np.random.permutation(sample_num)
    X = X[new_index]
    y = y[new_index]
    X_valid = X[:valid_num]
    y_valid = y[:valid_num]
    X_train = X[valid_num:]
    y_train = y[valid_num:]
    print('X_train.shape=', X_train.shape, 'y_train.shape=', y_train.shape)
    print('X_valid.shape=', X_valid.shape, 'y_valid.shape=', y_valid.shape)
    # 验证集打 batch
    print('creating batch data.')
    sample_num = len(X_valid)
    print('valid_sample_num=%d' % sample_num)
    train_batch(X_valid, y_valid, wd_valid_path, batch_size)
    # 训练集打 batch
    sample_num = len(X_train)
    print('train_sample_num=%d' % sample_num)
    train_batch(X_train, y_train, wd_train_path, batch_size)


def wd_test_get_batch(title_len=30, batch_size=128):
    print('loading word eval_title and eval_content.')
    eval_title = np.load('../data/wd_eval_title.npy')
    eval_content = np.load('../data/wd_eval_content.npy')
    p = Pool()
    X_title = np.asarray(p.map(pad_X30, eval_title))
    X_content = np.asarray(p.map(wd_pad_cut_docs, eval_content))
    p.close()
    p.join()
    X_content.shape = [-1, 30*10]
    X = np.hstack([X_title, X_content])
    sample_num = len(X)
    print('eval_sample_num=%d' % sample_num)
    eval_batch(X, wd_test_path, batch_size)


# char 数据打包
def ch_train_get_batch(title_len=52, batch_size=128):
    print('loading char train_title and train_content, this should cost minutes, please wait.')
    train_title = np.load('../data/ch_train_title.npy')
    train_content = np.load('../data/ch_train_content.npy')
    p = Pool(8)
    X_title = np.asarray(p.map(pad_X52, train_title))
    X_content = np.asarray(p.map(ch_pad_cut_docs, train_content))
    p.close()
    p.join()
    X_content.shape = [-1, 52*10]
    X = np.hstack([X_title, X_content])
    y = np.load('../data/y_tr.npy')
    # 划分验证集
    sample_num = X.shape[0]
    np.random.seed(13)
    valid_num = 100000
    new_index = np.random.permutation(sample_num)
    X = X[new_index]
    y = y[new_index]
    X_valid = X[:valid_num]
    y_valid = y[:valid_num]
    X_train = X[valid_num:]
    y_train = y[valid_num:]
    print('X_train.shape=', X_train.shape, 'y_train.shape=', y_train.shape)
    print('X_valid.shape=', X_valid.shape, 'y_valid.shape=', y_valid.shape)
    # 验证集打batch
    print('creating batch data.')
    sample_num = len(X_valid)
    print('valid_sample_num=%d' % sample_num)
    train_batch(X_valid, y_valid, ch_valid_path, batch_size)
    # 训练集打batch
    sample_num = len(X_train)
    print('train_sample_num=%d' % sample_num)
    train_batch(X_train, y_train, ch_train_path, batch_size)


def ch_test_get_batch(title_len=52, batch_size=128):
    print('loading char eval_title and eval_content.')
    eval_title = np.load('../data/ch_eval_title.npy')
    eval_content = np.load('../data/ch_eval_content.npy')
    p = Pool()
    X_title = np.asarray(p.map(pad_X52, eval_title))
    X_content = np.asarray(p.map(ch_pad_cut_docs, eval_content))
    p.close()
    p.join()
    X_content.shape = [-1, 52*10]
    X = np.hstack([X_title, X_content])
    sample_num = len(X)
    print('eval_sample_num=%d' % sample_num)
    eval_batch(X, ch_test_path, batch_size)


if __name__ == '__main__':
    wd_train_get_batch()
    wd_test_get_batch()
    ch_train_get_batch()
    ch_test_get_batch()