from __future__ import print_function
import numpy as np
import os
import sys
import random
from six.moves import cPickle as pickle


MOUSE_MOTIF_VARIANTS = [ 'AATAAA', 'ATTAAA', 'TATAAA', 'AGTAAA', 'AAGAAA', 'AATATA', 'AATACA', 'CATAAA', 'GATAAA', 'ACTAAA', 'AATAGA']

def get_data(data_root, label):
    data = []
    data_1 = []
    for data_file in os.listdir(data_root):
        data_1.clear()
        data_path = os.path.join(data_root, data_file)

        with open(data_path, 'r') as f:
            alphabet = np.array(['A', 'G', 'T', 'C'])
            for line in f:
                
                line = list(line.strip('\n'))
                seq = np.array(line, dtype = '|U1').reshape(-1, 1)

                seq_data = (seq == alphabet).astype(np.float32)

                data_1.append(seq_data)
                
            data_1 = data_1[:len(data_1)//10*9]
            for i in data_1:
                data.append(i)
    data = np.stack(data).reshape([-1, 206, 1, 4])
    if label:
        labels = np.zeros(data.shape[0])
    else:
        labels = np.ones(data.shape[0])
    return data, labels


def get_motif_data(data_root, label):
    data = {}
    labels = {}
    for motif in MOUSE_MOTIF_VARIANTS:
        data[motif] = []
        for data_file in os.listdir(data_root):
            if motif in data_file:
                data_path = os.path.join(data_root, data_file)
                with open(data_path, 'r') as f:
                    alphabet = np.array(['A', 'G', 'T', 'C'])
                    for line in f:
                        line = line.upper()
                        line = list(line.strip('\n'))
                        seq = np.array(line, dtype = '|U1').reshape(-1, 1)
                        seq_data = (seq == alphabet).astype(np.float32)
                        data[motif].append(seq_data)
        data[motif] = np.stack(data[motif]).reshape([-1, 206, 1, 4])
        if label:
            labels[motif] = np.zeros(data[motif].shape[0])
        else:
            labels[motif] = np.ones(data[motif].shape[0])
    return data, labels



def shuffle(dataset, labels, randomState=None):
    if randomState is None:
        permutation = np.random.permutation(labels.shape[0])
    else:
        permutation = randomState.permutation(labels.shape[0])
    shuffled_data = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_data, shuffled_labels


def data_split(pos_data, pos_labels, neg_data, neg_labels, num_folds, split, fold):
    pos_data_folds = np.array_split(pos_data, num_folds)
    neg_data_folds = np.array_split(neg_data, num_folds)
    pos_label_folds = np.array_split(pos_labels, num_folds)
    neg_label_folds = np.array_split(neg_labels, num_folds)

    train_pos_data = np.concatenate([pos_data_folds[i] for i in split['train']], axis=0)
    print(train_pos_data.shape)
    print('#####')
    train_pos_data_folds = np.array_split(train_pos_data, 10)
    train_pos_data = np.concatenate([train_pos_data_folds[i] for i in fold['round']], axis=0)
    print(train_pos_data.shape)
    train_pos_labels = np.concatenate([pos_label_folds[i] for i in split['train']], axis=0)
    train_pos_labels_folds = np.array_split(train_pos_labels, 10)
    train_pos_labels = np.concatenate([train_pos_labels_folds[i] for i in fold['round']], axis=0)
    
    valid_pos_data = np.concatenate([pos_data_folds[i] for i in split['valid']], axis=0)
    valid_pos_labels = np.concatenate([pos_label_folds[i] for i in split['valid']], axis=0)

    train_neg_data = np.concatenate([neg_data_folds[i] for i in split['train']], axis=0)
    train_neg_data_folds = np.array_split(train_neg_data, 10)
    train_neg_data = np.concatenate([train_neg_data_folds[i] for i in fold['round']], axis=0)
    

    
    train_neg_labels = np.concatenate([neg_label_folds[i] for i in split['train']], axis=0)
    train_neg_labels_folds = np.array_split(train_neg_labels, 10)
    train_neg_labels = np.concatenate([train_neg_labels_folds[i] for i in fold['round']], axis=0)
    valid_neg_data = np.concatenate([neg_data_folds[i] for i in split['valid']], axis=0)
    valid_neg_labels = np.concatenate([neg_label_folds[i] for i in split['valid']], axis=0)

    train_data = np.concatenate((train_pos_data, train_neg_data), axis=0)
    valid_data = np.concatenate((valid_pos_data, valid_neg_data), axis=0)
    train_labels = np.concatenate((train_pos_labels, train_neg_labels), axis=0)
    valid_labels = np.concatenate((valid_pos_labels, valid_neg_labels), axis=0)

    data = {}

    data['train_dataset'], data['train_labels'] = shuffle(train_data, train_labels)
    data['valid_dataset'], data['valid_labels'] = shuffle(valid_data, valid_labels)

    if 'test' in split:
        test_pos_data = np.concatenate([pos_data_folds[i] for i in split['test']], axis=0)
        test_pos_labels = np.concatenate([pos_label_folds[i] for i in split['test']], axis=0)
        test_neg_data = np.concatenate([neg_data_folds[i] for i in split['test']], axis=0)
        test_neg_labels = np.concatenate([neg_label_folds[i] for i in split['test']], axis=0)
        test_data = np.concatenate((test_pos_data, test_neg_data), axis=0)
        test_labels = np.concatenate((test_pos_labels, test_neg_labels), axis=0)
        data['test_dataset'], data['test_labels'] = shuffle(test_data, test_labels)

    return data


def produce_dataset(num_folds, pos_path, neg_path, seed=0):
    pos_data, pos_labels = get_data(pos_path, True)
    neg_data, neg_labels = get_data(neg_path, False)
    randomState = np.random.RandomState(seed)
    pos_data, pos_labels = shuffle(pos_data, pos_labels, randomState)
    neg_data, neg_labels = shuffle(neg_data, neg_labels, randomState)
    print('Positive:', pos_data.shape, pos_labels.shape)
    print('Negative:', neg_data.shape, neg_labels.shape)
    return pos_data, pos_labels, neg_data, neg_labels


