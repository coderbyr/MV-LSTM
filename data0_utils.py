"""This is the Data Utils for Letor source code.

This module is used to read data from letor dataset.
"""

__version__ = '0.2'
__author__ = 'Lipengyu'

import sys
import random
import json
import numpy as np
import pytextnet as pt


config = json.loads(open(sys.argv[1]).read())
data_path = config['data_dir']

word_dict, iword_dict = pt.io.base.read_word_dict(filename=data_path + '/mp_word_dict.txt')
query_data = pt.io.base.read_data(filename=data_path + '/qid_mp.txt')
doc_data = pt.io.base.read_data(filename=data_path + '/docid_mp.txt')

class PairGenerator():
    def __init__(self, mention_embedding_file, entity_embedding_file, rel_file, config):
        self.train_mention, self.train_entity, self.test_mention, self.test_entity = \
            pt.io.base.read_embedding(mention_embedding_file, entity_embedding_file)
        self.rel = pt.io.base.read_relation(filename=rel_file)
        self.config = config
        self.l = len(self.rel)
        self.idx = []

    def get_batch(self):
        self.idx = [x for x in range(len(self.rel))]
        random.shuffle(self.idx)
        for start_idx in range(0, len(self.idx) - config['batch_size'] + 1, config['batch_size']):
            excerpt = slice(start_idx, start_idx + config['batch_size'])
            yield self.idx[excerpt]

    def get_batch2(self, idx_list):
        pair_list = [self.rel[i] for i in idx_list]
        train_mention = self.train_mention[idx_list]
        train_entity = self.train_entity[idx_list]
        config = self.config
        X1 = np.zeros((config['batch_size'], config['data1_maxlen']), dtype=np.int32)
        X2 = np.zeros((config['batch_size'], config['data2_maxlen']), dtype=np.int32)
        Y = np.zeros((config['batch_size'], 2), dtype=np.float32)

        embed1 = np.zeros((config['batch_size'], config['data_maxlen'], config['embed_size']), dtype=np.float32)
        embed2 = np.zeros((config['batch_size'], config['data_maxlen'], config['embed_size']), dtype=np.float32)

        for idx, val in enumerate(pair_list):
            label, entity, mention = val
            embed1[idx] = train_mention[idx]
            embed2[idx] = train_entity[idx]
            X1[idx, :] = mention
            X2[idx, :] = entity
            if label == 0:
                Y[idx][0] = 1
            else:
                Y[idx][1] = 1
        return X1, X2, Y, embed1, embed2

class ListGenerator():
    def __init__(self, mention_embedding_file, entity_embedding_file, rel_file, config):
        self.train_mention, self.train_entity, self.test_mention, self.test_entity = \
            pt.io.base.read_embedding(mention_embedding_file, entity_embedding_file)
        self.rel = pt.io.base.read_relation(filename=rel_file)
        self.config = config
        self.l = len(self.rel)
        self.idx = [x for x in range(self.l)]

    def get_batch2(self):
        random.shuffle(self.idx)
        idx_list = self.idx
        pair_list = [self.rel[i] for i in idx_list]
        test_mention = self.test_mention[idx_list]
        test_entity = self.test_entity[idx_list]

        config = self.config
        X1 = np.zeros((self.l, config['data1_maxlen']), dtype=np.int32)
        X2 = np.zeros((self.l, config['data2_maxlen']), dtype=np.int32)
        Y = np.zeros((self.l, 2), dtype=np.float32)

        embed1 = np.zeros((self.l, config['data_maxlen'], config['embed_size']), dtype=np.float32)
        embed2 = np.zeros((self.l, config['data_maxlen'], config['embed_size']), dtype=np.float32)

        # X1[:] = config['fill_word']
        # X2[:] = config['fill_word']
        for idx, val in enumerate(pair_list):
            label, mention, entity = val
            embed1[idx] = test_mention[idx]
            embed2[idx] = test_entity[idx]
            X1[idx, :] = mention
            X2[idx, :] = entity

            if label == 0:
                Y[idx][0] = 1
            else:
                Y[idx][1] = 1
        return X1, X2, Y, embed1, embed2