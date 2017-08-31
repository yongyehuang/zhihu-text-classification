## 数据处理

1.把比赛提供的所有数据解压到 raw_data/ 目录下。
2.按照顺序依次执行各个 .py，不带任何参数。
  或者在当前目录下输入下面命令运行所有文件：
  dos2unix run_all_data_process.sh   # 使用cygwin工具dos2unix将script改为unix格式
  sh run_all_data_process.sh
3.环境依赖(下面是我使用的版本)
- numpy		1.12.1
- pandas 	0.19.2
- word2vec	0.9.1
- tqdm		4.11.2


### embed2ndarray.py
赛方提供了txt格式的词向量和字向量，这里把embedding矩阵转成 np.ndarray 形式，分别保存为 data/word_embedding.npy 和 data/char_embedding.npy。
用 pd.Series 保存词(字)对应 embedding 中的行号(id),存储在 data/sr_word2id.pkl 和 data/sr_char2id.pkl 中。

### question_and_topic_2id.py
把问题和话题转为id形式，保存在 data/sr_question2id.pkl 和 data/sr_id2question.pkl 中。

### char2id.py
利用上面得到的 sr_char2id，把所有问题的字转为对应的id, 存储为
data/ch_train_title.npy
data/ch_train_content.npy
data/ch_eval_title.npy
data/ch_eval_content.npy

### word2id.py
同 char2id.py

### creat_batch_data.py
把所有的数据按照 batch_size(128) 进行打包，固定seed，随机取 10 万样本作为验证集。每个batch存储为一个 npz 文件，包括 X, y 两部分。
这里所有的序列都进行了截断，长度不足的用0进行padding到固定长度。
保存位置：
wd_train_path = '../data/wd-data/data_train/'
wd_valid_path = '../data/wd-data/data_valid/'
wd_test_path = '../data/wd-data/data_test/'
ch_train_path = '../data/ch-data/data_train/'
ch_valid_path = '../data/ch-data/data_valid/'
ch_test_path = '../data/ch-data/data_test/'


### creat_batch_seg.py
和 creat_batch_data.py 相同，只是对 content 部分进行句子划分。用于分层模型。
划分句子长度：
wd_title_len = 30, wd_sent_len = 30, wd_doc_len = 10.(即content划分为10个句子，每个句子长度为30个词)
ch_title_len = 52, ch_sent_len = 52, ch_doc_len = 10.
不划分句子：
wd_title_len = 30, wd_content_len = 150.
ch_title_len = 52, ch_content_len = 300.


### To do
在这里所有的序列都截断或者padding为固定长度，这样子不光时间慢，而且在误差计算中没有处理padding部分，可能会是准确率下降。后期将改成使用 mask 和 dynamic_rnn处理padding部分。