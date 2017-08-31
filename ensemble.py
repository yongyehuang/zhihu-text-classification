
# coding: utf-8

# ## 线上测试集模型融合

# In[1]:

from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import pickle
import os
import sys
import time 


# 求 softmax
def _softmax(score):
    """对一个样本的输出类别概率进行 softmax 归一化.
    score: arr.shape=[1999].
    """
    max_sc = np.max(score)   # 最大分数
    score = score - max_sc
    exp_sc = np.exp(score)
    sum_exp_sc = np.sum(exp_sc)
    softmax_sc = exp_sc / sum_exp_sc
    return softmax_sc    # 归一化的结果
    
def softmax(scores):
    """对所有样本的输出概率进行 softmax 归一化处理。
    scores: arr.shape=[n_sample, 1999].
    """
    softmax_scs = map(_softmax, scores)
    return np.asarray(softmax_scs)


# ### 模型求加权平均
# 这里的所有模型以及相应的权重是通过 local-ensemble 中对线下验证集进行权重调整得到的最好结果。

# In[3]:

time0 = time.time()
scores_names =[
    'p1-1-bigru-512.npy',
    'p1-2-bigru-512-true.npy',
    'textcnn-fc-drop-title-content-256-3457-drop0.5.npy',
    'f1-1-cnn-256-23457-11.npy',

    'han-cnn-title-content-256-345.npy',
    'han-cnn-title-content-256-23457-1234.npy',
    'm7-rnn-cnn-256-100.npy',

    'p3-2-cnn-256-2357.npy',
    'p3-cnn-512-23457.npy',
    'textcnn-fc-drop-title-content-256-345.npy',   
    'textcnn-fc-drop-title-content-256-3457-drop0.2.npy',

    'm9-han-bigru-title-content-512-30.npy',
    'm9-2-han-bigru-title-content-512-30.npy',
    'han-bigru-title-content-256-30.npy',
    'm8-han-bigru-256-30.npy',

    'attention-bigru-title-content-256.npy',
    'm7-2-rnn-cnn-128-100.npy',
    'textcnn-fc-title-content-256-345.npy',
    'm1-2-fasttext-topicinfo.npy',

    'ch3-1-cnn-256-2345.npy',
    'ch3-2-cnn-256-23457.npy', 
    'ch4-1-han-bigru-256-52.npy',    
    'ch5-1-2embed-rnn256-cnn2345.npy',

    'p4-1-han-bigru-256.npy',
    'ch6-1-han-cnn-2345-1234.npy',
    'p5-1-2embed-rnn256-cnn2345.npy',
    'ch5-2-2embed-rnn512-cnn3457.npy',

    'c1-1-cnn-max-256-23457.npy',
    'c1-2-cnn-256-345710.npy',     
    'c2-1-bigru-256.npy',
    
    'textcnn-fc-drop-title-content-256-345-cross3cross0.npy',
    'textcnn-fc-drop-title-content-256-345-cross3cross1.npy',
    'textcnn-fc-drop-title-content-256-345-cross3cross2.npy',
    'p3-3-cnn-max-256-345710.npy',
    'textcnn-title-256-len50.npy',
    'ch7-1-2embed-rnn256-hcnn-2345-1234.npy',

#    'p2-1-rnn-cnn-256-256.npy',
]  

# weights = [  9.75938817,   8.63945014,  2.98289344,   3.72323394,   5.04378259,
#   0.06551187,  -0.79412528,   4.90162676,   1.17452791,
#  -1.46124679,  -0.25384273,   5.50925013,   2.84186738,  -0.93016907,
#   5.16519035,  -0.47061662,   2.75998217,   2.58152296,  -1.24553333,
#   2.43288558,   6.17376317,   5.59323762,  10.46123521,   5.29952925,
#   3.72042086,   5.46707444,   5.51516916,   5.82352659,   1.27847427,
#  -0.52930247,  -1.99052155,  -3.0938045,   -2.07007845,   4.19963813,
#   2.10593832,   1.74174258, -0.21665029]

weights = [  9.75938817,   8.63945014,  2.98289344,   3.72323394,   5.04378259,
   0.06551187,  -0.79412528,   4.90162676,   1.17452791,
  -1.46124679,  -0.25384273,   5.50925013,   2.84186738,  -0.93016907,
   5.16519035,  -0.47061662,   2.75998217,   2.58152296,  -1.24553333,
   2.43288558,   6.17376317,   5.59323762,  10.46123521,   5.29952925,
   3.72042086,   5.46707444,   5.51516916,   5.82352659,   1.27847427,
  -0.52930247,  -1.99052155,  -3.0938045,   -2.07007845,   4.19963813,
   2.10593832,   1.74174258]


print(len(scores_names), len(weights))
print('All %d models' % len(weights))
sum_scores = np.zeros((217360, 1999), dtype=float)
scores_path = 'scores/'
for i in xrange(len(weights)):
    scores_name = scores_names[i]
    print('%d/%d, scores_name=%s' %(i+1, len(weights),scores_name))
    score = np.load(scores_path + scores_name)
    score = softmax(score) # 加归一化
    sum_scores = sum_scores + score* weights[i]
print('sum_scores.shape=',sum_scores.shape)
print('Finished , costed time %g s' % (time.time() - time0))



# 写入 result
result_path = 'ye-final36-result.csv'

def write_result(sum_scores, result_path):
    """把结果写到 sum_result.csv 中"""
    print('Begin computing...')
    predict_labels_list = map(lambda label: label.argsort()[-1:-6:-1], sum_scores) # 取最大的5个下标
    eval_question = np.load('data/eval_question.npy')
    with open('data/sr_topic2id.pkl', 'rb') as inp:
        sr_topic2id = pickle.load(inp)
        sr_id2topic = pickle.load(inp)
    pred_labels = np.asarray(predict_labels_list).reshape([-1])
    pred_topics = sr_id2topic[pred_labels].values.reshape([-1, 5])   # 转为 topic
    df_result = pd.DataFrame({'question':eval_question, 'tid0': pred_topics[:,0], 'tid1':pred_topics[:, 1],
                         'tid2': pred_topics[:,2], 'tid3':pred_topics[:,3],'tid4': pred_topics[:,4]})
    df_result.to_csv(result_path, index=False, header=False)
    print('Finished writing the result')
    return df_result

time0 = time.time()
write_result(sum_scores, result_path)
print('Result path %s, costed time %g s' % (result_path, time.time() - time0))

