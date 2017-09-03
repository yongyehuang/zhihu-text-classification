# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

"""wd_4_han
title 部分使用 bigru+attention；content 部分使用 han； 两部分输出直接 concat。
"""


class Settings(object):
    def __init__(self):
        self.model_name = 'wd_4_han'
        self.title_len = self.sent_len = 30
        self.doc_len = 10
        self.hidden_size = 256
        self.n_layer = 1
        self.fc_hidden_size = 1024
        self.n_class = 1999
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'


class HAN(object):
    """
    title: inputs->bigru+attention->output_title
    content: inputs->sent_encoder(bigru+attention)->doc_encoder(bigru+attention)->output_content
    concat[output_title, output_content] -> fc+bn+relu -> sigmoid_entropy.
    """

    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.title_len = self.sent_len = settings.sent_len
        self.doc_len = settings.doc_len
        self.hidden_size = settings.hidden_size
        self.n_layer = settings.n_layer
        self.n_class = settings.n_class
        self.fc_hidden_size = settings.fc_hidden_size
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        # placeholders
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            self._X1_inputs = tf.placeholder(tf.int64, [None, self.title_len], name='X1_inputs')
            self._X2_inputs = tf.placeholder(tf.int64, [None, self.doc_len * self.sent_len], name='X2_inputs')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('bigru_text'):
            output_title = self.bigru_inference(self._X1_inputs)

        with tf.variable_scope('han_content'):
            output_content = self.han_inference(self._X2_inputs)

        with tf.variable_scope('fc-bn-layer'):
            output = tf.concat([output_title, output_content], axis=1)
            W_fc = self.weight_variable([self.hidden_size * 4, self.fc_hidden_size], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")
            fc_bn_drop = tf.nn.dropout(self.fc_bn_relu, self.keep_prob)

        with tf.variable_scope('out_layer'):
            W_out = self.weight_variable([self.fc_hidden_size, self.n_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.n_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self._y_pred = tf.nn.xw_plus_b(fc_bn_drop, W_out, b_out, name='y_pred')  # 每个类别的分数 scores

        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self._y_pred, labels=self._y_inputs))
            tf.summary.scalar('loss', self._loss)

        self.saver = tf.train.Saver(max_to_keep=1)

    @property
    def tst(self):
        return self._tst

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def global_step(self):
        return self._global_step

    @property
    def X1_inputs(self):
        return self._X1_inputs

    @property
    def X2_inputs(self):
        return self._X2_inputs

    @property
    def y_inputs(self):
        return self._y_inputs

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def loss(self):
        return self._loss

    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def batchnorm(self, Ylogits, offset, convolutional=False):
        """batchnormalization.
        Args:
            Ylogits: 1D向量或者是3D的卷积结果。
            num_updates: 迭代的global_step
            offset：表示beta，全局均值；在 RELU 激活中一般初始化为 0.1。
            scale：表示lambda，全局方差；在 sigmoid 激活中需要，这 RELU 激活中作用不大。
            m: 表示batch均值；v:表示batch方差。
            bnepsilon：一个很小的浮点数，防止除以 0.
        Returns:
            Ybn: 和 Ylogits 的维度一样，就是经过 Batch Normalization 处理的结果。
            update_moving_everages：更新mean和variance，主要是给最后的 test 使用。
        """
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, self._global_step)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(self.tst, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(self.tst, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    def gru_cell(self):
        with tf.name_scope('gru_cell'):
            cell = rnn.GRUCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bi_gru(self, inputs, seg_num):
        """build the bi-GRU network. Return the encoder represented vector.
        n_step: 句子的词数量；或者文档的句子数。
        seg_num: 序列的数量，原本应该为 batch_size, 但是这里将 batch_size 个 doc展开成很多个句子。
        """
        cells_fw = [self.gru_cell() for _ in range(self.n_layer)]
        cells_bw = [self.gru_cell() for _ in range(self.n_layer)]
        initial_states_fw = [cell_fw.zero_state(seg_num, tf.float32) for cell_fw in cells_fw]
        initial_states_bw = [cell_bw.zero_state(seg_num, tf.float32) for cell_bw in cells_bw]
        outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                        initial_states_fw = initial_states_fw, initial_states_bw = initial_states_bw, dtype=tf.float32)
        # outputs: Output Tensor shaped: seg_num, max_time, layers_output]，其中layers_output=hidden_size * 2 在这里。
        return outputs

    def task_specific_attention(self, inputs, output_size,
                                initializer=layers.xavier_initializer(),
                                activation_fn=tf.tanh, scope=None):
        """
        Performs task-specific attention reduction, using learned
        attention context vector (constant within task of interest).
        Args:
            inputs: Tensor of shape [batch_size, units, input_size]
                `input_size` must be static (known)
                `units` axis will be attended over (reduced from output)
                `batch_size` will be preserved
            output_size: Size of output's inner (feature) dimension
        Returns:
           outputs: Tensor of shape [batch_size, output_dim].
        """
        assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
        with tf.variable_scope(scope or 'attention') as scope:
            # u_w, attention 向量
            attention_context_vector = tf.get_variable(name='attention_context_vector', shape=[output_size],
                                                       initializer=initializer, dtype=tf.float32)
            # 全连接层，把 h_i 转为 u_i ， shape= [batch_size, units, input_size] -> [batch_size, units, output_size]
            input_projection = layers.fully_connected(inputs, output_size, activation_fn=activation_fn, scope=scope)
            # 输出 [batch_size, units]
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(vector_attn, dim=1)
            tf.summary.histogram('attention_weigths', attention_weights)
            weighted_projection = tf.multiply(inputs, attention_weights)
            outputs = tf.reduce_sum(weighted_projection, axis=1)
            return outputs  # 输出 [batch_size, hidden_size*2]

    def bigru_inference(self, X_inputs):
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        output_bigru = self.bi_gru(inputs, self.batch_size)
        output_att = self.task_specific_attention(output_bigru, self.hidden_size*2)
        return output_att   # 输出 [batch_size, hidden_size*2]

    def han_inference(self, X_inputs):
        """分层 attention 模型。content部分。
        Args:
            X_inputs: tensor.shape=(batch_size, doc_len*sent_len)
        Returns:
            doc_attn_outputs: tensor.shape=(batch_size, hidden_size(*2 for bigru))
        """
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)    # inputs.shape=[batch_size, doc_len*sent_len, embedding_size]
        sent_inputs = tf.reshape(inputs,[self.batch_size*self.doc_len, self.sent_len, self.embedding_size]) # shape=(?, 40, 256)
        with tf.variable_scope('sentence_encoder'):  # 生成句向量
            sent_outputs = self.bi_gru(sent_inputs, seg_num=self.batch_size*self.doc_len)
            sent_attn_outputs = self.task_specific_attention(sent_outputs, self.hidden_size*2) # [batch_size*doc_len, hidden_size*2]
            with tf.variable_scope('dropout'):
                sent_attn_outputs = tf.nn.dropout(sent_attn_outputs, self.keep_prob)
        with tf.variable_scope('doc_encoder'):      # 生成文档向量
            doc_inputs = tf.reshape(sent_attn_outputs, [self.batch_size, self.doc_len, self.hidden_size*2])
            doc_outputs = self.bi_gru(doc_inputs, self.batch_size)  # [batch_size, doc_len, hidden_size*2]
            doc_attn_outputs = self.task_specific_attention(doc_outputs, self.hidden_size*2) # [batch_size, hidden_size*2]
        return doc_attn_outputs    # [batch_size, hidden_size*2]



# test the model
def test():
    import numpy as np
    print('Begin testing...')
    settings = Settings()
    W_embedding = np.random.randn(50, 10)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    batch_size = 128
    with tf.Session(config=config) as sess:
        model = HAN(W_embedding, settings)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(model.loss)
        update_op = tf.group(*model.update_emas)
        sess.run(tf.global_variables_initializer())
        fetch = [model.loss, model.y_pred, train_op, update_op]
        loss_list = list()
        for i in xrange(100):
            X1_batch = np.zeros((batch_size, 30), dtype=float)
            X2_batch = np.zeros((batch_size, 10 * 30), dtype=float)
            y_batch = np.zeros((batch_size, 1999), dtype=int)
            _batch_size = len(y_batch)
            feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch, model.y_inputs: y_batch,
                         model.batch_size: _batch_size, model.tst: False, model.keep_prob: 0.5}
            loss, y_pred, _, _ = sess.run(fetch, feed_dict=feed_dict)
            loss_list.append(loss)
            print(i, loss)

if __name__ == '__main__':
    test()
