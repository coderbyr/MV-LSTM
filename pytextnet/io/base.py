# encoding: utf-8
# author: lipengyu

import json
import numpy as np

# Read Embedding File
def read_embedding(mention_embedding, entity_embedding, train_size=56516):
    mention = np.load(mention_embedding)
    entity = np.load(entity_embedding)
    train_mention = mention[:train_size]
    train_entity = entity[:train_size]
    test_mention = mention[train_size:]
    test_entity = entity[train_size:]
    #print '[%s]\n\t[%s]\n\tEmbedding size: %d' % (mention_embedding, entity_embedding, len(mention))
    return train_mention, train_entity, test_mention, test_entity

# Read Relation Data
def read_relation(filename):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) != 3:
                continue
            data.append((int(line[0]), line[1], line[2]))
    return data


# Read Word Dict and Inverse Word Dict
def read_word_dict(filename):
    word_dict = {}
    iword_dict = {}
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) != 2:
                continue
            word_dict[int(line[1])] = line[0]
            iword_dict[line[0]] = int(line[1])
    print '[%s]\n\tWord dict size: %d' % (filename, len(word_dict))
    return word_dict, iword_dict


def read_data(filename):
    data = {}
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) < 2:
                continue
            if len(line[2:]) != int(line[1]):
                continue
            data[line[0]] = map(int, line[2:])
    print '[%s]\n\tData size: %s' % (filename, len(data))
    return data