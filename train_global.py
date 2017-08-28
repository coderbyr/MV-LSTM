"""This is the Training Structure source code.

This module is the main function of model training.
usage:
    python train_global.py [config_file]
"""

__version__ = '0.2'
__author__ = 'Lipengyu'

import re
import os
import sys
import json
import random
import time
import data0_utils as du
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from importlib import import_module

sys.path.insert(0, 'model/')

config = json.loads(open(sys.argv[1]).read())
word_dict = du.word_dict
entity_dict = du.query_data
mention_dict = du.doc_data

train_gen = du.TrainGenerator(mention_embedding_file=config['mention_embedd_path'],
                            entity_embedding_file=config['entity_embedd_path'],
                            rel_file=config['training_path'], config=config)

mo = import_module(config['model_file'])
model = mo.Model(config)

sess = tf.Session()
model.init_step(sess)

#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

def savePreditions(word_dict, X1, X2, Y, idx_list):
    of1 = open(config['right_path'], 'w')
    of2 = open(config['err_path'], 'w')
    for i in range(len(X1)):
        m1, e1 = X1[i], X2[i]
        m1 = mention_dict[str(m1[0])]
        e1 = entity_dict[str(e1[0])]
        m1_len, e1_len = len(m1), len(e1)
        label = Y[i].tolist().index(1)
        if label == 1:
            if idx_list[i]:
                pred_label = 1
            else:
                pred_label = 0
        else:
            if idx_list[i]:
                pred_label = 0
            else:
                pred_label = 1
        m1, e1 = m1[:m1_len], e1[:e1_len]
        m1 = map(lambda r: word_dict[r] if r in word_dict else "", m1)
        e1 = map(lambda r: word_dict[r] if r in word_dict else "", e1)
        m1, e1 = "".join(m1), "".join(e1)
        if idx_list[i]:
            of1.write(str(label) + "\t" + str(pred_label) + "\t" + m1 + "\t" + e1 + "\n")
        else:
            of2.write(str(label) + "\t" + str(pred_label) + "\t" + m1 + "\t" + e1 + "\n")

    of1.close()
    of2.close()

#flog = open(config['log_file'], 'w')

for epoch in range(1, config['num_epochs']+1):
    print("Epoch %d ..." % epoch)

    train_err = 0.0
    train_corr = 0.0
    train_total = 0.0
    start_time = time.time()
    train_batches = 0
    batch_count = int(train_gen.l/config['batch_size'])

    #record session summary
    train_writer = tf.summary.FileWriter(config['log_path'] + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(config['log_path'] + '/validation')
    test_writer = tf.summary.FileWriter(config['log_path'] + '/test')
    merged = tf.summary.merge_all()

    for train_list in train_gen.get_batch():
        X1, X2, Y, embed1, embed2 = train_gen.get_batch_data(train_list)
        feed_dict = { model.X1: X1, model.X2: X2, model.Y: Y,
                      model.embed1: embed1, model.embed2: embed2, model.keep_prob: 0.5}
        num = len(X1)

        loss, accu, summary = model.train_step(sess, merged, feed_dict)
        train_err += loss
        train_corr += num*accu
        train_batches += 1
        train_total += num
        train_writer.add_summary(summary, epoch * batch_count + train_batches)
        # if train_batches%10 == 0:
        #     print "train %d batch, time %s" % (train_batches, time.time() - start_time)
    print('train: %d/%d loss: %.4f, acc: %.2f%%, time: %.2fs' % (
         train_batches * config['batch_size'], train_gen.l,
         train_err, train_corr * 100 / train_total, time.time() - start_time))

    ##validation 
    test_gen = du.TestGenerator(mention_embedding_file=config['mention_embedd_path'],
                                entity_embedding_file=config['entity_embedd_path'],
                                rel_file=config['validation_path'],
                                config=config)
    val_err = 0.0
    val_corr = 0.0
    val_total = 0.0
    start_time = time.time()
    val_num = 0
    X1, X2, Y, embed1, embed2 = test_gen.get_test_data()
    feed_dict = {model.X1: X1, model.X2: X2, model.Y: Y,
                 model.embed1: embed1, model.embed2: embed2, model.keep_prob: 0.8}
    num = len(X1)
    val_loss, val_accu, precision, recall, f1, idx_list, summary = model.test_step(sess, merged, feed_dict)
    val_err += val_loss
    val_corr += num*val_accu
    val_total += num
    validation_writer.add_summary(summary, epoch)
    if epoch == config['num_epochs']:
        savePreditions(word_dict, X1, X2, Y, idx_list)
    print('validation: %d, loss: %.4f, acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1-score: %.2f%%, time: %.2fs' % (
         val_total, val_err, val_corr * 100 / val_total, precision*100, recall*100, f1*100, time.time() - start_time))

    test_gen = du.TestGenerator(mention_embedding_file=config['mention_embedd_path'],
                                entity_embedding_file=config['entity_embedd_path'],
                                rel_file=config['test_path'],
                                config=config)
    test_err = 0.0
    test_corr = 0.0
    test_total = 0.0
    start_time = time.time()
    test_num = 0
    X1, X2, Y, embed1, embed2 = test_gen.get_test_data()
    feed_dict = {model.X1: X1, model.X2: X2, model.Y: Y,
                 model.embed1: embed1, model.embed2: embed2, model.keep_prob: 0.8}
    num = len(X1)
    test_loss, test_accu, precision, recall, f1, idx_list, summary = model.test_step(sess, merged, feed_dict)
    test_err += test_loss
    test_corr += num*test_accu
    test_total += num
    test_writer.add_summary(summary, epoch)
    print('test: %d, loss: %.4f, acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1-score: %.2f%%, time: %.2fs' % (
         test_total, test_err, test_corr * 100 / test_total, precision*100, recall*100, f1*100, time.time() - start_time))




