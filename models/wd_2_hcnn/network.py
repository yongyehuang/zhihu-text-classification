# -*- coding:utf-8 -*-

import tensorflow as tf

"""wd_2_hcnn
title 部分使用 TextCNN；content 部分使用分层的 TextCNN。
"""


class Settings(object):
    def __init__(self):
        self.model_name = 'wd_2_hcnn'
        self.title_len = self.sent_len = 30
        self.doc_len = 10
        self.sent_filter_sizes = [2, 3, 4, 5]
        self.doc_filter_sizes = [2, 3, 4]
        self.n_filter = 256
        self.fc_hidden_size = 1024
        self.n_class = 1999
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'


class HCNN(object):
    """
    title: inputs->textcnn->output_title
    content: inputs->hcnn->output_content
    concat[output_title, output_content] -> fc+bn+relu -> sigmoid_entropy.
    """

    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.sent_len = settings.sent_len
        self.doc_len = settings.doc_len
        self.sent_filter_sizes = settings.sent_filter_sizes
        self.doc_filter_sizes = settings.doc_filter_sizes
        self.n_filter = settings.n_filter
        self.n_class = settings.n_class
        self.fc_hidden_size = settings.fc_hidden_size
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        # placeholders
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            self._X1_inputs = tf.placeholder(tf.int64, [None, self.sent_len], name='X1_inputs')
            self._X2_inputs = tf.placeholder(tf.int64, [None, self.doc_len * self.sent_len], name='X2_inputs')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('cnn_text'):
            output_title = self.cnn_inference(self._X1_inputs)

        with tf.variable_scope('hcnn_content'):
            output_content = self.hcnn_inference(self._X2_inputs)

        with tf.variable_scope('fc-bn-layer'):
            output = tf.concat([output_title, output_content], axis=1)
            output_size = self.n_filter * (len(self.sent_filter_sizes) + len(self.doc_filter_sizes))
            W_fc = self.weight_variable([output_size, self.fc_hidden_size], name='Weight_fc')
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

        self.saver = tf.train.Saver(max_to_keep=2)

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
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                           self._global_step)  # adding the iteration prevents from averaging across non-existing iterations
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

    def textcnn(self, X_inputs, n_step, filter_sizes, embed_size):
        """build the TextCNN network.
        n_step: the sentence len."""
        inputs = tf.expand_dims(X_inputs, -1)
        pooled_outputs = list()
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embed_size, 1, self.n_filter]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter")
                beta = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.n_filter], name="beta"))
                tf.summary.histogram('beta', beta)
                conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv_bn, update_ema = self.batchnorm(conv, beta, convolutional=True)  # 在激活层前面加 BN
                # Apply nonlinearity, batch norm scaling is not useful with relus
                # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
                h = tf.nn.relu(conv_bn, name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
                self.update_emas.append(update_ema)
        h_pool = tf.concat(pooled_outputs, 3)
        n_filter_total = self.n_filter * len(filter_sizes)
        h_pool_flat = tf.reshape(h_pool, [-1, n_filter_total])
        return h_pool_flat  # shape = [-1, n_filter_total]

    def cnn_inference(self, X_inputs):
        """TextCNN 模型。title部分。
        Args:
            X_inputs: tensor.shape=(batch_size, title_len)
        Returns:
            title_outputs: tensor.shape=(batch_size, n_filter*filter_num_sent)
        """
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        with tf.variable_scope('title_encoder'):  # 生成 title 的向量表示
            title_outputs = self.textcnn(inputs, self.sent_len, self.sent_filter_sizes, embed_size=self.embedding_size)
        return title_outputs  # shape = [batch_size, n_filter*filter_num_sent]

    def hcnn_inference(self, X_inputs):
        """分层 TextCNN 模型。content部分。
        Args:
            X_inputs: tensor.shape=(batch_size, doc_len*sent_len)
        Returns:
            doc_attn_outputs: tensor.shape=(batch_size, n_filter*filter_num_doc)
        """
        inputs = tf.nn.embedding_lookup(self.embedding,
                                        X_inputs)  # inputs.shape=[batch_size, doc_len*sent_len, embedding_size]
        sent_inputs = tf.reshape(inputs, [self.batch_size * self.doc_len, self.sent_len,
                                          self.embedding_size])  # [batch_size*doc_len, sent_len, embedding_size]
        with tf.variable_scope('sentence_encoder'):  # 生成句向量
            sent_outputs = self.textcnn(sent_inputs, self.sent_len, self.sent_filter_sizes, self.embedding_size)
        with tf.variable_scope('doc_encoder'):  # 生成文档向量
            doc_inputs = tf.reshape(sent_outputs, [self.batch_size, self.doc_len, self.n_filter * len(
                self.sent_filter_sizes)])  # [batch_size, doc_len, n_filter*len(filter_sizes_sent)]
            doc_outputs = self.textcnn(doc_inputs, self.doc_len, self.doc_filter_sizes, self.n_filter * len(
                self.sent_filter_sizes))  # [batch_size, doc_len, n_filter*filter_num_doc]
        return doc_outputs  # [batch_size,  n_filter*len(doc_filter_sizes)]

# test the model
# def test():
#     import numpy as np
#     print('Begin testing...')
#     settings = Settings()
#     W_embedding = np.random.randn(50, 10)
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     batch_size = 128
#     with tf.Session(config=config) as sess:
#         model = HCNN(W_embedding, settings)
#         optimizer = tf.train.AdamOptimizer(0.001)
#         train_op = optimizer.minimize(model.loss)
#         update_op = tf.group(*model.update_emas)
#         sess.run(tf.global_variables_initializer())
#         fetch = [model.loss, model.y_pred, train_op, update_op]
#         loss_list = list()
#         for i in xrange(100):
#             X1_batch = np.zeros((batch_size, 30), dtype=float)
#             X2_batch = np.zeros((batch_size, 10 * 30), dtype=float)
#             y_batch = np.zeros((batch_size, 1999), dtype=int)
#             _batch_size = len(y_batch)
#             feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch, model.y_inputs: y_batch,
#                          model.batch_size: _batch_size, model.tst: False, model.keep_prob: 0.5}
#             loss, y_pred, _, _ = sess.run(fetch, feed_dict=feed_dict)
#             loss_list.append(loss)
#             print(i, loss)

# test()
