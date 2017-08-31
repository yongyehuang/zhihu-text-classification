# -*- coding:utf-8 -*- 

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import word2vec
import pickle
import os


def get_word_embedding():
    """提取词向量，并保存至 ../data/word_embedding.npy"""
    print('getting the word_embedding.npy')
    wv = word2vec.load('../raw_data/word_embedding.txt')
    word_embedding = wv.vectors
    words = wv.vocab
    sr_id2word = pd.Series(words, index=range(1,1+len(words)))
    sr_word2id = pd.Series(range(1,1+len(words)), index=words)
    # 添加 'UNKNOWN' 符号进行 padding ,将填充字符编号为 0
    embedding_size = 256
    sr_id2word[0] = 'UNKNOWN'
    sr_word2id['UNKNOWN'] = 0
    mean_embedding = np.mean(word_embedding)
    pad_vector = np.zeros([1, embedding_size]) + mean_embedding
    word_embedding = np.vstack([pad_vector, word_embedding])
    # 保存词向量
    save_path = '../data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path+'word_embedding.npy', word_embedding)
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
    sr_id2char = pd.Series(chars, index=range(1,1+len(chars)))
    sr_char2id = pd.Series(range(1,1+len(chars)), index=chars)
    # 添加 'UNKNOWN' 符号进行 padding ,将填充字符编号为 0
    embedding_size = 256
    sr_id2char[0] = 'UNKNOWN'
    sr_char2id['UNKNOWN'] = 0
    mean_embedding = np.mean(char_embedding)
    pad_vector = np.zeros([1, embedding_size]) + mean_embedding
    char_embedding = np.vstack([pad_vector, char_embedding])
    # 保存字向量
    save_path = '../data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path+'char_embedding.npy', char_embedding)
    # 保存字与id的对应关系
    with open(save_path + 'sr_char2id.pkl', 'wb') as outp:
        pickle.dump(sr_id2char, outp)
        pickle.dump(sr_char2id, outp)
    print('Saving the char_embedding.npy to ../data/char_embedding.npy')    
        

if __name__ == '__main__':
    get_word_embedding()
    get_char_embedding()

