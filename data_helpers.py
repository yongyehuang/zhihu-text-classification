#!usr/bin/python 
# -*- coding:utf-8 -*- 

"""
Construct a Data generator.
"""
import numpy as np
from tqdm import tqdm
import os


class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.
    
    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=True)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """ 
    
    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]
                
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_examples(self):
        return self._number_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data 
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]
    
    
def to_categorical(topics):
    """把所有的topic id 转为 0，1形式。
    Args:
        topics: n_sample 个 lists, 问题的话题标签。每个list对应一个问题，topic个数不定。
    return:
        y: ndarray, shape=(sample， n_class)， 其中 n_class = 1999.
    Example:
     >>> y_batch = to_categorical(topic_batch)
     >>> print(y_batch.shape)
     >>> (10, 1999)
    """
    n_sample = len(topics)
    y = np.zeros(shape=(n_sample, 1999))
    for i in xrange(n_sample):
        topic_index = topics[i]
        y[i, topic_index] = 1
    return y


def pad_X30(words, max_len=30):
    """把 word_ids 整理成固定长度。
    """
    words_len = len(words)
    words = np.asarray(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[:max_len]
    return np.hstack([words, np.zeros(max_len-words_len, dtype=int)])


def pad_X50(words, max_len=50):
    """把 word_ids 整理成固定长度。
    """
    words_len = len(words)
    words = np.asarray(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[:max_len]
    return np.hstack([words, np.zeros(max_len-words_len, dtype=int)])


def pad_X52(words, max_len=52):
    """把 word_ids 整理成固定长度。
    """
    words_len = len(words)
    words = np.asarray(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[:max_len]
    return np.hstack([words, np.zeros(max_len-words_len, dtype=int)])


def pad_X150(words, max_len=150):
    """把 word_ids 整理成固定长度。
    """
    words_len = len(words)
    words = np.asarray(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[:max_len]
    return np.hstack([words, np.zeros(max_len-words_len, dtype=int)])

def pad_X300(words, max_len=300):
    """把 word_ids 整理成固定长度。
    """
    words_len = len(words)
    words = np.asarray(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[:max_len]
    return np.hstack([words, np.zeros(max_len-words_len, dtype=int)])


def wd_cut_docs(words_id, max_len=30):
    """
    把 doc 切割成句子。如果句子长度超过 max_len, 将句子按照最长 max_len切割成多个句子。
    Args:
        words_id: list or np.array, 整个文档的词对应的 id，[ 2336, 1468, 69, 49241, 68, 5 ... ]
        max_len: 切割后最长的句子长度。
    Returns:
        segs: list of list，每个元素为一个list，即一个句子。 每个元素list包含多个id.
    """
    if type(words_id) is np.ndarray:
        words_id = words_id.tolist()
    if type(words_id) is not list:
        print('Type error! the words_id should be list or numpy.ndarray')
    set_splits = set([5, 6, 50])      # 切割符号所对应的id
    ws_len = len(words_id)
    cut_index = filter(lambda i: words_id[i] in set_splits, range(len(words_id)))
    segs = list()                        # 分割之后的句子
    if len(cut_index) == 0:                  # 如果没有切割符号
        seg_len = len(words_id)
        if seg_len > max_len:                # 如果超长，切割
            for start in xrange(0, seg_len, max_len):
                end = min(seg_len, start+max_len)
                segs.append(words_id[start:end])
        else:                         # 否则，整个句子返回
            segs.append(words_id)
        return segs
    if cut_index[-1] != ws_len - 1:   # 最后一个不是切割符号
        cut_index = cut_index + [ws_len-1]
    cut_index = np.asarray(cut_index) + 1
    cut_index = cut_index.tolist()
    start = [0] + cut_index[:-1]
    end = cut_index
    cut_indexs = zip(start, end)
    for index in cut_indexs:
        if index[1] == index[0]:                 # 1.如果第一个就是分割符号，去掉
            continue
        seg_len = index[1] - index[0]
        if seg_len == 1:                        # 2.如果只有一个词
            if words_id[index[0]] not in set_splits:    # 并且不是分隔符
                segs.append([words_id[index[0]]])     # 那么添加
            continue                            # 否则丢弃
        if seg_len > max_len:                   # 3.如果超长，切割
            for start in xrange(index[0], index[1], max_len):
                end = min(index[1], start+max_len)
                segs.append(words_id[start:end])
        else:
            segs.append(words_id[index[0]:index[1]])  # 4.添加序列
    return segs


def wd_pad_cut_docs(words_id, doc_len=10, sent_len=30):
    """把 doc 切割成句子，并 padding 成固定个句子数，每个句子长度也固定为 sent_len.
    Args:
        words_id: list or np.array, 整个文档的词对应的 id，[ 2336, 1468, 69, 49241, 68, 5 ... ]
        doc_len: int, 每个文档的句子数，超过 doc_len 的丢弃；少于 doc_len 的补全。
        sent_len: int, 每个句子的最大长度， 不足 sent_len 的使用 0 (id for 'UNKNOWN')进行补全。
    Returns:
        segs: np.adarray, shape=[doc_len, sent_len].
    """
    segs4doc = wd_cut_docs(words_id, max_len=sent_len)
    segs4doc = np.asarray(map(pad_X30, segs4doc))    # 每一部分都进行补齐
    segs_num = segs4doc.shape[0]  # 当前句子数
    if segs_num >= doc_len:       # 如果句子数够了
        return segs4doc[:doc_len, :]
    if segs_num == 0:
        return np.zeros(shape=(doc_len, sent_len), dtype=int)
    segs4doc = np.vstack([segs4doc, np.zeros(shape=(doc_len-segs_num, sent_len), dtype=int)])
    return segs4doc


def ch_cut_docs(chs_id, max_len=52):
    """
    把 doc 切割成句子。如果句子长度超过 max_len, 将句子按照最长 max_len切割成多个句子。
    Args:
        chs_id: list or np.array, 整个文档的字对应的 id，[ 2336, 1468, 69, 49241, 68, 5 ... ]
        max_len: 切割后最长的句子长度。
    Returns:
        segs: list of list，每个元素为一个list，即一个句子。 每个元素list包含多个id.
    """
    if type(chs_id) is np.ndarray:
        chs_id = chs_id.tolist()
    if type(chs_id) is not list:
        print('Type error! the chs_id should be list or numpy.ndarray')
    set_splits = set([8, 14, 77])      # 切割符号所对应的id
    chs_len = len(chs_id)
    cut_index = filter(lambda i: chs_id[i] in set_splits, range(len(chs_id)))
    segs = list()                     # 分割之后的句子
    if len(cut_index) == 0:           # 如果没有切割符号
        seg_len = len(chs_id)
        if seg_len > max_len:   # 如果超长，切割
            for start in xrange(0, seg_len, max_len):
                end = min(seg_len, start+max_len)
                segs.append(chs_id[start:end])
        else:                         # 否则，整个句子返回
            segs.append(chs_id)
        return segs
    if cut_index[-1] != chs_len - 1:   # 最后一个不是切割符号
        cut_index = cut_index + [chs_len-1]
    cut_index = np.asarray(cut_index) + 1
    cut_index = cut_index.tolist()
    start = [0] + cut_index[:-1]
    end = cut_index
    cut_indexs = zip(start, end)
    for index in cut_indexs:
        if index[1] == index[0]:                 # 1.如果第一个就是分割符号，去掉
            continue
        seg_len = index[1] - index[0]
        if seg_len == 1:                        # 2.如果只有一个词
            if chs_id[index[0]] not in set_splits:  # 并且不是分隔符
                segs.append([chs_id[index[0]]])     # 那么添加
            continue                            # 否则丢弃
        if seg_len > max_len:                   # 3.如果超长，切割
            for start in xrange(index[0], index[1], max_len):
                end = min(index[1], start+max_len)
                segs.append(chs_id[start:end])
        else:
            segs.append(chs_id[index[0]:index[1]])  # 4.添加序列
    return segs


def ch_pad_cut_docs(chs_id, doc_len=10, sent_len=52):
    """把 doc 切割成句子，并 padding 成固定个句子数，每个句子长度也固定为 sent_len.
    Args:
        chs_id: list or np.array, 整个文档的词对应的 id，[ 2336, 1468, 69, 49241, 68, 5 ... ]
        doc_len: int, 每个文档的句子数，超过 doc_len 的丢弃；少于 doc_len 的补全。
        sent_len: int, 每个句子的最大长度， 不足 sent_len 的使用 0 (id for 'UNKNOWN')进行补全。
    Returns:
        segs: np.adarray, shape=[doc_len, sent_len].
    """
    segs4doc = ch_cut_docs(chs_id, max_len=sent_len)
    segs4doc = np.asarray(map(pad_X52, segs4doc))      # 每一部分都进行补齐
    segs_num = segs4doc.shape[0]  # 当前句子数
    if segs_num >= doc_len:       # 如果句子数够了
        return segs4doc[:doc_len, :]
    if segs_num == 0:
        return np.zeros(shape=(doc_len, sent_len), dtype=int)
    segs4doc = np.vstack([segs4doc, np.zeros(shape=(doc_len-segs_num, sent_len), dtype=int)])
    return segs4doc


def train_batch(X, y, batch_path, batch_size=128):
    """对训练集打batch."""
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    sample_num = len(X)
    batch_num = 0
    for start in tqdm(xrange(0, sample_num, batch_size)):
        end = min(start + batch_size, sample_num)
        batch_name = batch_path + str(batch_num) + '.npz'
        X_batch = X[start:end]
        y_batch = y[start:end]
        np.savez(batch_name, X=X_batch, y=y_batch)
        batch_num += 1
    print('Finished, batch_num=%d' % (batch_num+1))


def eval_batch(X, batch_path, batch_size=128):
    """对测试数据打batch."""
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    sample_num = len(X)
    print('sample_num=%d' % sample_num)
    batch_num = 0
    for start in tqdm(xrange(0, sample_num, batch_size)):
        end = min(start + batch_size, sample_num)
        batch_name = batch_path + str(batch_num) + '.npy'
        X_batch = X[start:end]
        np.save(batch_name, X_batch)
        batch_num += 1
    print('Finished, batch_num=%d' % (batch_num+1))