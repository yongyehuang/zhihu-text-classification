# -*- coding:utf-8 -*- 

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import word2vec
import pickle
import os

SPECIAL_SYMBOL = ['<PAD>', '<EOS>']  # add these special symbols to word(char) embeddings.


def get_word_embedding():
    """提取词向量，并保存至 ../data/word_embedding.npy"""
    print('getting the word_embedding.npy')
    wv = word2vec.load('../raw_data/word_embedding.txt')
    word_embedding = wv.vectors
    words = wv.vocab
    sr_id2word = pd.Series(words, index=range(1, 1 + len(words)))
    sr_word2id = pd.Series(range(1, 1 + len(words)), index=words)
    # 添加特殊符号：<PAD>:0, <UNK>:1
    embedding_size = 256
    n_special_sym = len(SPECIAL_SYMBOL)
    vec_special_sym = np.random.randn(n_special_sym, embedding_size)
    for i in range(n_special_sym):
        sr_id2word[i] = SPECIAL_SYMBOL[i]
        sr_word2id[SPECIAL_SYMBOL[i]] = i
    word_embedding = np.vstack([vec_special_sym, word_embedding])
    # 保存词向量
    save_path = '../data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + 'word_embedding.npy', word_embedding)
    # 保存词与id的对应关系
    with open(save_path + 'sr_word2id.pkl', 'wb') as outp:
        pickle.dump(sr_id2word, outp)
        pickle.dump(sr_word2id, outp)
    print('Saving the word_embedding.npy to ../data/word_embedding.npy')


def get_char_embedding():
    """提取字向量，并保存至 ../data/char_embedding.npy"""
    print('getting the char_embedding.npy')
    wv = word2vec.load('../raw_data/char_embedding.txt')
    char_embedding = wv.vectors
    chars = wv.vocab
    sr_id2char = pd.Series(chars, index=range(1, 1 + len(chars)))
    sr_char2id = pd.Series(range(1, 1 + len(chars)), index=chars)

    # 添加特殊符号：<PAD>:0, <UNK>:1
    embedding_size = 256
    n_special_sym = len(SPECIAL_SYMBOL)
    vec_special_sym = np.random.randn(n_special_sym, embedding_size)
    for i in range(n_special_sym):
        sr_id2char[i] = SPECIAL_SYMBOL[i]
        sr_char2id[SPECIAL_SYMBOL[i]] = i
    char_embedding = np.vstack([vec_special_sym, char_embedding])
    # 保存字向量
    save_path = '../data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + 'char_embedding.npy', char_embedding)
    # 保存字与id的对应关系
    with open(save_path + 'sr_char2id.pkl', 'wb') as outp:
        pickle.dump(sr_id2char, outp)
        pickle.dump(sr_char2id, outp)
    print('Saving the char_embedding.npy to ../data/char_embedding.npy')


if __name__ == '__main__':
    get_word_embedding()
    get_char_embedding()
