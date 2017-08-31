# 2017 知乎看山杯 多标签文本分类
## 1.运行环境
下面是我实验中的一些环境依赖，版本只提供参考。

|环境/库|版本|
|:---------:|----------|
|Ubuntu|14.04.5 LTS|
|python|2.7.12|
|jupyter notebook|4.2.3|
|tensorflow-gpu|1.2.1|
|numpy|1.12.1|
|pandas|0.19.2|
|matplotlib|2.0.0|
|word2vec|0.9.1|
|tqdm|4.11.2|

## 2.文件结构

|- zhihu-text-classification<br/>
|　　|- raw_data　　　　　　　　　# 比赛提供的原始数据<br/>
|　　|- data　　　　　　　　　　　# 预处理得到的数据<br/>
|　　|- data_process　　　　　　　# 数据预处理代码<br/>
|　　|- models　　　　　　　　　　# 模型代码<br/>
|　　|　　|- wd-1-1-cnn-concat<br/>
|　　|　　|　　|- network.py　　　　　　# 定义网络结构<br/>
|　　|　　|　　|- train.py　　　　　　  # 模型训练<br/>
|　　|　　|　　|- predict.py　　　　　　# 验证集/测试集预测，生成概率矩阵<br/>
...<br/>
|　　|- ckpt　　　　　　　　　　　# 保存训练好的模型<br/>
|　　|- summary　　　　　　　　　　# tensorboard数据<br/>
|　　|- scores　　　　　　　　　　　# 测试集的预测概率矩阵<br/>
|　　|- local_scores　　　　　　　 # 验证集的预测概率矩阵<br/>
|　　|- doc　　　　　　　　　　    # 文档说明与相关论文<br/>
|　　|- notebook-old　　　　　　　# 比赛中未经过整理的代码<br/>
|　　|- local_ensemble.ipynb　　　# 验证集模型融合<br/>
|　　|- ensemble.py　　　　　　　　# 测试集模型融合<br/>
|　　|- data_helpers.py　　　　　　# 数据处理函数<br/>
|　　|- evaluator.py　　　　　　　 # 评价函数<br/>


## 3.数据预处理
- 把比赛提供的所有数据解压到 raw_data/ 目录下。
- 按照顺序依次执行下面每个 .py，不带任何参数。<br/>
  或者在当前目录下输入下面命令运行所有文件：<br/>
  dos2unix run_all_data_process.sh   # 使用cygwin工具dos2unix将script改为unix格式<br/>
  sh run_all_data_process.sh

### 3.1 embed2ndarray.py
赛方提供了txt格式的词向量和字向量，这里把embedding矩阵转成 np.ndarray 形式，分别保存为 data/word_embedding.npy 和 data/char_embedding.npy。
用 pd.Series 保存词(字)对应 embedding 中的行号(id),存储在 data/sr_word2id.pkl 和 data/sr_char2id.pkl 中。

### 3.2 question_and_topic_2id.py
把问题和话题转为id形式，保存在 data/sr_question2id.pkl 和 data/sr_id2question.pkl 中。

### 3.3 char2id.py
利用上面得到的 sr_char2id，把所有问题的字转为对应的id, 存储为<br/>
data/ch_train_title.npy<br/>
data/ch_train_content.npy<br/>
data/ch_eval_title.npy<br/>
data/ch_eval_content.npy<br/>

### 3.4 word2id.py
同 char2id.py

### 3.5 creat_batch_data.py
把所有的数据按照 batch_size(128) 进行打包(显存大的可以取256)，固定seed，随机取 10 万样本作为验证集。每个batch存储为一个 npz 文件，包括 X, y 两部分。
这里所有的序列都进行了截断，长度不足的用0进行padding到固定长度。<br/>
保存位置：<br/>
wd_train_path = '../data/wd-data/data_train/'<br/>
wd_valid_path = '../data/wd-data/data_valid/'<br/>
wd_test_path = '../data/wd-data/data_test/'<br/>
ch_train_path = '../data/ch-data/data_train/'<br/>
ch_valid_path = '../data/ch-data/data_valid/'<br/>
ch_test_path = '../data/ch-data/data_test/'<br/>


### 3.6 creat_batch_seg.py
和 creat_batch_data.py 相同，只是对 content 部分进行句子划分。用于分层模型。
划分句子长度：<br/>
wd_title_len = 30, wd_sent_len = 30, wd_doc_len = 10.(即content划分为10个句子，每个句子长度为30个词)<br/>
ch_title_len = 52, ch_sent_len = 52, ch_doc_len = 10.<br/>
不划分句子：<br/>
wd_title_len = 30, wd_content_len = 150.<br/>
ch_title_len = 52, ch_content_len = 300.<br/>


## 4.模型训练
切换到模型所在位置，然后进行训练和预测。比如：
```
cd zhihu-text-classification/models/wd-1-1-cnn-concat/
# 训练
python train.py [--max_epoch 1 --max_max_epoch 6 --lr 1e-3 decay_rate 0.65 decay_step 15000 last_f1 0.4]
# 预测
python predict.py
```
这里只整理了部分模型，所有模型都用的词向量。如果想要使用字向量，只需要把模型中的输入和序列长度修改即可。

## 5.模型融合
线性加权融合，模拟梯度下降的策略进行权值搜索。见：local_ensemble.ipynb
注意：
- 此方法可能会对验证集过拟合，所以需要通过测试集进一步判断。在模型个数比较多时使用此方法效果更好。
- 需要根据各个单模型的性能认为进行初始化。char 和 word 类型不能直接比较，char 的单模型的性能虽然较差，但是对融合提升非常明显。