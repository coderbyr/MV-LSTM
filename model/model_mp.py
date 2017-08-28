"""
This is the Model File of MV-LSTM.
"""

import sys
import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_bug 

"""
Model Class
"""
class Model():
    def __init__(self, config):
        self.config = config
        tf.reset_default_graph()
        self.X1 = tf.placeholder(tf.int32, name='X1', shape=(None, config['data1_maxlen']))
        self.X2 = tf.placeholder(tf.int32, name='X2', shape=(None, config['data2_maxlen']))
        self.Y = tf.placeholder(tf.float32, name='Y', shape=(None, 2))
        self.keep_prob = tf.placeholder(tf.float32)

        self.k_num = config['k_num']
        self.output_dim = config['output_dim']
        self.embed_size = config['embed_size']

        self.batch_size = tf.shape(self.X1)[0]

        self.embed1 = tf.placeholder(tf.float32, name='embedding1', shape=(None, config['data_maxlen'], self.embed_size))
        self.embed2 = tf.placeholder(tf.float32, name='embedding2', shape=(None, config['data_maxlen'], self.embed_size))

        # construct neural tensor layer
        self.w1 = tf.get_variable('w1', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32) , dtype=tf.float32, shape=[self.output_dim, self.embed_size, self.embed_size])
        self.v1 = tf.get_variable('v1', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32) , dtype=tf.float32, shape=[2*self.embed_size, config['data_maxlen'], self.output_dim])

        self.feed_forward_output = tf.einsum('abd,def->abef', tf.concat([self.embed1, self.embed2], 2), self.v1)

        # batch_size * 20 * 20
        self.blinear_tensor_output = []

        for i in range(self.output_dim):
            # batch _size * 10 * 40
            self.f0 = tf.einsum('abc,cd->abd', self.embed1, self.w1[i])

            # batch_size * 10 * 10
            self.f1 = tf.einsum('abc,acd->abd', self.embed2, tf.transpose(self.f0, perm=[0, 2, 1]))
            # batch_size * 10 * 10
            self.blinear_tensor_output.append(self.f1)

        # batch_size * 10 * 10 * 32
        self.tensor_output = tf.nn.relu(tf.reshape(tf.concat(self.blinear_tensor_output, axis=0),
                                              (self.batch_size, self.config['data_maxlen'], self.config['data_maxlen'],
                                               self.output_dim))
                                   + self.feed_forward_output)
        #self.tensor_output = tf.placeholder(tf.float32, name="tensor_output", shape=(None, config['data_maxlen'], config['data_maxlen'], self.output_dim))


        # k-max pooling
        self.k_max_pooling = self.make_sparse_layer(self.tensor_output, self.k_num, self.config)

        # MLP
        self.w2 = tf.get_variable('w2', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, shape=[self.k_num * self.output_dim, 64])
        self.b2 = tf.get_variable('b2', initializer=tf.constant_initializer(), dtype=tf.float32, shape=[64])
        self.fc1 = tf.nn.relu(tf.matmul(self.k_max_pooling, self.w2) + self.b2)

        self.w3 = tf.get_variable('w3', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, shape=[64, 32])
        self.b3 = tf.get_variable('b3', initializer=tf.constant_initializer(), dtype=tf.float32, shape=[32])
        self.fc2 = tf.nn.relu(tf.matmul(self.fc1, self.w3) + self.b3)
        self.fc2_drop = tf.nn.dropout(self.fc2, self.keep_prob)

        self.w4 = tf.get_variable('w4', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, shape=[32, 2])
        self.b4 = tf.get_variable('b4', initializer=tf.constant_initializer(), dtype=tf.float32, shape=[2])
        self.pred = tf.nn.softmax(tf.matmul(self.fc2_drop, self.w4) + self.b4)

        # cross entropy as loss function
        with tf.name_scope('Loss'):
            self.loss = tf.reduce_mean(
                -tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(self.pred, 1e-10, 1.0)), reduction_indices=[1]))

        self.predictions, self.actuals = tf.argmax(self.pred, 1), tf.argmax(self.Y, 1)

        # save predictions and labels
        self.idx = tf.equal(self.predictions, self.actuals)

        self.train_model = tf.train.AdamOptimizer().minimize(self.loss)
        self.correct_pre = tf.equal(self.predictions, self.actuals)

        self.saver = tf.train.Saver(max_to_keep=20)

        # compute accuracy, precision, recall, f1-score
        ones_like_actuals = tf.ones_like(self.actuals)
        zeros_like_actuals = tf.zeros_like(self.actuals)
        ones_like_predictions = tf.ones_like(self.predictions)
        zeros_like_predictions = tf.zeros_like(self.predictions)

        self.TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, ones_like_actuals),
                                                       tf.equal(self.predictions, ones_like_predictions)), 'float'))
        self.TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, zeros_like_actuals),
                                                       tf.equal(self.predictions, zeros_like_predictions)), 'float'))
        self.FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, zeros_like_actuals),
                                                       tf.equal(self.predictions, ones_like_predictions)), 'float'))
        self.FN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(self.actuals, ones_like_actuals),
                                                       tf.equal(self.predictions, zeros_like_predictions)), 'float'))

        self.precision = self.TP * 1.0 / (self.TP + self.FP)
        self.recall = self.TP * 1.0 / (self.TP + self.FN)
        self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        with tf.name_scope('Accu'):
            self.accu = (self.TP + self.TN) * 1.0 / (self.TP + self.TN + self.FP + self.FN)

        #add tensorboard code
        tf.summary.histogram('w1', self.w1)
        tf.summary.histogram('b1', self.v1)
        tf.summary.histogram('w2', self.w1)
        tf.summary.histogram('b2', self.b2)
        tf.summary.histogram('w3', self.w1)
        tf.summary.histogram('b3', self.b3)
        tf.summary.histogram('w4', self.w1)
        tf.summary.histogram('b4', self.b4)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar('Accu', self.accu)

    def make_sparse_layer(self, input_x, k, config):
        #input_x = tf.convert_to_tensor(input_x, dtype=tf.float32)
        d = input_x.get_shape().as_list()[0]
        matrix_in = tf.reshape(input_x, [-1, config['data_maxlen']*config['data_maxlen']])
        values, indices = tf.nn.top_k(matrix_in, k=k, sorted=False)
        res = tf.reshape(values, [-1, self.output_dim*k])
        return res

    def make_tensor_output(self, sess, embed1, embed2):
        # batch_size * 10 * 10 * 32
        embed1 = tf.convert_to_tensor(embed1, dtype=tf.float32)
        embed2 = tf.convert_to_tensor(embed2, dtype=tf.float32)
        feed_forward_output = tf.einsum('abd,def->abef', tf.concat([embed1, embed2], 2), self.v1)
        batch_size = embed1.get_shape().as_list()[0]

        #batch_size * 20 * 20
        blinear_tensor_output = []

        for i in range(self.output_dim):
            # batch _size * 10 * 40
            f0 = tf.einsum('abc,cd->abd', embed1, self.w1[i])

            # batch_size * 10 * 10
            f1 = tf.einsum('abc,acd->abd', embed2, tf.transpose(f0, perm=[0, 2, 1]))
            #batch_size * 10 * 10
            blinear_tensor_output.append(f1)

        # batch_size * 10 * 10 * 32
        tensor_output = tf.nn.relu(tf.reshape(tf.concat(blinear_tensor_output, axis=0),
                                                   (batch_size, self.config['data_maxlen'], self.config['data_maxlen'], self.output_dim))
                                                    + feed_forward_output)
        return sess.run(tensor_output)

    def init_step(self, sess):
        sess.run(tf.global_variables_initializer())

    def train_step(self, sess, merged, feed_dict):
        _, train_accu, loss, summary = sess.run([self.train_model, self.accu, self.loss, merged],
                                       feed_dict=feed_dict)
        return loss, train_accu, summary

    def test_step(self, sess, merged, feed_dict):
        _, test_accu, loss, precision, recall, F1, idx, summary = sess.run([self.train_model, self.accu, self.loss, self.precision, self.recall, self.F1, self.idx, merged],
                                                  feed_dict=feed_dict)
        return loss, test_accu, precision, recall, F1, idx, summary

