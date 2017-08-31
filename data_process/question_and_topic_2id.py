# -*- coding:utf-8 -*- 

import pandas as pd
import pickle
from itertools import chain


def question_and_topic_2id():
    """把question和topic转成id形式并保存至 ../data/目录下。"""
    print('Changing the quetion and topic to id and save in sr_question2.pkl and sr_topic2id.pkl in ../data/')
    df_question_topic = pd.read_csv('../raw_data/question_topic_train_set.txt', sep='\t', names=['question', 'topics'],
                        dtype={'question': object, 'topics': object})
    df_question_topic.topics = df_question_topic.topics.apply(lambda tps: tps.split(','))
    save_path = '../data/'
    print('questino number = %d ' % len(df_question_topic))
    # 问题 id 按照给出的问题顺序编号
    questions = df_question_topic.question.values
    sr_question2id = pd.Series(range(len(questions)), index=questions) 
    sr_id2question = pd.Series(questions, index=range(len(questions)))
    # topic 按照数量从大到小进行编号
    topics = df_question_topic.topics.values
    topics = list(chain(*topics))
    sr_topics = pd.Series(topics)
    topics_count = sr_topics.value_counts()
    topics = topics_count.index
    sr_topic2id = pd.Series(range(len(topics)),index=topics)
    sr_id2topic = pd.Series(topics, index=range(len(topics))) 

    with open(save_path + 'sr_question2id.pkl', 'wb') as outp:
        pickle.dump(sr_question2id, outp)
        pickle.dump(sr_id2question, outp)
    with open(save_path + 'sr_topic2id.pkl', 'wb') as outp:
        pickle.dump(sr_topic2id, outp)
        pickle.dump(sr_id2topic, outp)
    print('Finished changing.')


if __name__ == '__main__':
    question_and_topic_2id()
