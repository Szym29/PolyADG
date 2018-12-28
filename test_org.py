#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import sys, os
from decimal import Decimal
from hm_prep import *
############ Model Selection ############
POS_PATH = 'data/human/dragon_polyA_data/positive5fold/'
NEG_PATH = 'data/human/dragon_polyA_data/negatives5fold/'
#POS_PATH = 'data/human/omni_polyA_data/positive/'
#NEG_PATH = 'data/human/omni_polyA_data/negative/'
BATCH_SIZE = 64
PATCH_SIZE = 10
DEPTH = 16
NUM_HIDDEN = 64
SEQ_LEN = 206 + 2*PATCH_SIZE-2
NUM_CHANNELS = 4
NUM_LABELS = 2
NUM_EPOCHS = 200
NUM_FOLDS = 5
HYPER_DICT = True
Hex = True
############ **************** ############

tf.app.flags.DEFINE_string(
    'train_dir', None,
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer(
    'training_job_index', 0,
    'index of training result for logging')

tf.app.flags.DEFINE_string(
    'training_result_dir', None,
    'The file which the training result is written to')

FLAGS = tf.app.flags.FLAGS


def pad_dataset(dataset, labels):
    ''' Change dataset height to height + 2*DEPTH - 2'''
    new_dataset = np.ones([dataset.shape[0], dataset.shape[1]+2*PATCH_SIZE-2, dataset.shape[2], dataset.shape[3]], dtype = np.float32) * 0.25
    new_dataset[:, PATCH_SIZE-1:-(PATCH_SIZE-1), :, :] = dataset
    labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
    return new_dataset, labels


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def gen_hyper_dict(hyper_dict=None):
    def rand_log(a, b):
        x = np.random.sample()
        return 10.0 ** ((np.log10(b) - np.log10(a)) * x + np.log10(a))

    def rand_sqrt(a, b):
        x = np.random.sample()
        return (b - a) * np.sqrt(x) + a

    if hyper_dict is None:
        hyper_dict = {
            'tf_learning_rate': rand_log(.0005, .05),
            'tf_momentum': rand_sqrt(.95, .99),
            'tf_motif_init_weight': rand_log(1e-2, 10),
            'tf_fc_init_weight': rand_log(1e-2, 10),
            'tf_motif_weight_decay': rand_log(1e-5, 1e-3),
            'tf_fc_weight_decay': rand_log(1e-5, 1e-3),
            'tf_keep_prob': np.random.choice([.5, .75, 1.0]),
            'tf_ngroups': np.random.choice([2,4,8]),


        }
    if hyper_dict is not None:
        hyper_dict = pickle.load(open(os.path.join('/home/zzym/polyA_HEX/org_hyper', 'result.pkl'),'rb'))
    print(hyper_dict)
    # for k, v in hyper_dict.items():
    #     print("%s: %.2e"%(k, Decimal(v)))
    # print()
    return hyper_dict


# Disable print
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore print
def enable_print():
    sys.stdout = sys.__stdout__
def test(dataset, hyper_dict, num):
    graph = tf.Graph()

    with graph.as_default():
        # Load hyper-params
        tf_learning_rate = hyper_dict['tf_learning_rate']
        tf_momentum = hyper_dict['tf_momentum']
        tf_motif_init_weight = hyper_dict['tf_motif_init_weight']
        tf_fc_init_weight = hyper_dict['tf_fc_init_weight']
        tf_motif_weight_decay = hyper_dict['tf_motif_weight_decay']
        tf_fc_weight_decay = hyper_dict['tf_fc_weight_decay']
        tf_keep_prob = hyper_dict['tf_keep_prob']
        tf_ngroups = hyper_dict['tf_ngroups']

        # Input data.
        tf_test_dataset = tf.constant(np.concatenate([dataset['train_dataset'],dataset['valid_dataset'],dataset['test_dataset']]))
        print(tf_test_dataset.shape)
        tf_test_label = tf.constant(np.concatenate([dataset['train_labels'],dataset['valid_labels'],dataset['test_labels']]))
        tf_motif_test_dataset = {}
        tf_motif_test_label = {}
        for motif in HUMAN_MOTIF_VARIANTS:
            tf_motif_test_dataset[motif] = tf.constant(np.concatenate([dataset['motif_dataset'][motif]['train_dataset'],dataset['motif_dataset'][motif]['valid_dataset'],dataset['motif_dataset'][motif]['test_dataset']]))
            tf_motif_test_label[motif] = tf.constant(np.concatenate([dataset['motif_dataset'][motif]['train_labels'],dataset['motif_dataset'][motif]['valid_labels'],dataset['motif_dataset'][motif]['test_labels']]))


        # Variables.
        conv_weights = tf.Variable(tf.truncated_normal(
            [PATCH_SIZE, 1, NUM_CHANNELS, DEPTH], stddev=tf_motif_init_weight))
        conv_biases = tf.Variable(tf.zeros([DEPTH]))
        layer1_weights = tf.Variable(tf.truncated_normal(
            [21 * DEPTH, NUM_HIDDEN], stddev=tf_fc_init_weight))
        layer1_biases = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [NUM_HIDDEN, NUM_LABELS], stddev=tf_fc_init_weight))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))



        w = pickle.load(open(os.path.join('/home/zzym/polyA_HEX/org_weights','cv%d_model.pkl'%num),'rb'))
        print('chekchehf',len(w))
        conv_weights = tf.Variable(tf.cast(w['conv_weights'],dtype=tf.float32))
        conv_biases = tf.Variable(tf.cast(w['conv_biases'],dtype=tf.float32))
        layer1_weights = tf.Variable(tf.cast(w['layer1_weights'],dtype=tf.float32))
        layer1_biases = tf.Variable(tf.cast(w['layer1_biases'],dtype=tf.float32))
        layer2_weights =tf.Variable(tf.cast(w['layer2_weights'],dtype=tf.float32))
        layer2_biases = tf.Variable(tf.cast(w['layer2_biases'],dtype=tf.float32))

        # Store Variables
        weights = {}
        weights['conv_weights'] = conv_weights
        weights['conv_biases'] = conv_biases
        weights['layer1_weights'] = layer1_weights
        weights['layer1_biases'] = layer1_biases
        weights['layer2_weights'] = layer2_weights
        weights['layer2_biases'] = layer2_biases


        # Model.
        def model(data,drop=True):
            conv = tf.nn.conv2d(data, conv_weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.reshape(conv, [-1, 215, 1, DEPTH // tf_ngroups, tf_ngroups])
            mu, var = tf.nn.moments(conv, [1, 2, 3], keep_dims=True)
            conv = (conv - mu) / tf.sqrt(var + 1e-12)
            conv = tf.reshape(conv, [-1, 215, 1, DEPTH])
            hidden = tf.nn.relu(conv + conv_biases)
            hidden = tf.nn.max_pool(hidden, [1, 10, 1, 1], [1, 10, 1, 1], padding='VALID')
            shape = hidden.get_shape().as_list()
            motif_score = tf.reshape(hidden, [shape[0], shape[1] * DEPTH])
            if drop:
                hidden_nodes = tf.nn.dropout(tf.nn.relu(tf.matmul(motif_score, layer1_weights) + layer1_biases),
                                             tf_keep_prob)
            else:
                hidden_nodes = tf.nn.relu(tf.matmul(motif_score, layer1_weights) + layer1_biases)
            return tf.matmul(hidden_nodes, layer2_weights) + layer2_biases


        # Predictions for the training, validation, and test data.

        test_prediction = tf.nn.softmax(model(tf_test_dataset, drop=False))
        motif_test_prediction = {}
        for motif in HUMAN_MOTIF_VARIANTS:
            motif_test_prediction[motif] = tf.nn.softmax(model(tf_motif_test_dataset[motif], drop=False))

    test_results = []
    motif_test_results = {motif: [] for motif in HUMAN_MOTIF_VARIANTS}

    with tf.Session(graph=graph) as session, tf.device('/gpu:0'):
        tf.global_variables_initializer().run()
        test_pred = test_prediction.eval()
        test_results.append(accuracy(test_pred, np.concatenate([dataset['train_labels'],dataset['valid_labels'],dataset['test_labels']])))
        for motif in HUMAN_MOTIF_VARIANTS:
            motif_test_pred = motif_test_prediction[motif].eval()
            motif_test_results[motif].append(accuracy(motif_test_pred, np.concatenate([dataset['motif_dataset'][motif]['train_labels'],dataset['motif_dataset'][motif]['valid_labels'],dataset['motif_dataset'][motif]['test_labels']])))

    return  test_results,  motif_test_results
def main(_):

    # block_print()

    hyper_dict = gen_hyper_dict(HYPER_DICT)
    pos_data, pos_labels, neg_data, neg_labels = produce_motif_dataset(NUM_FOLDS, POS_PATH, NEG_PATH)
    test_accuracy_split = []
    motif_test_accuracy_split = {motif: [] for motif in HUMAN_MOTIF_VARIANTS}

    for i in range(0,5):
        split =  {
            'train': [(i + j) % NUM_FOLDS for j in range(NUM_FOLDS-2)],
            'valid': [(i + NUM_FOLDS-2) % NUM_FOLDS],
            'test': [(i + NUM_FOLDS-1) % NUM_FOLDS]
            }
        save = motif_data_split(pos_data, pos_labels, neg_data, neg_labels, NUM_FOLDS, split)
        dataset = {}
        dataset['train_dataset'], dataset['train_labels'] = pad_dataset(save['train_dataset'], save['train_labels'])
        dataset['valid_dataset'], dataset['valid_labels'] = pad_dataset(save['valid_dataset'], save['valid_labels'])
        dataset['test_dataset'], dataset['test_labels'] = pad_dataset(save['test_dataset'], save['test_labels'])

        dataset['motif_dataset'] = {}
        for motif in HUMAN_MOTIF_VARIANTS:
            dataset['motif_dataset'][motif] = {}
            dataset['motif_dataset'][motif]['test_dataset'], dataset['motif_dataset'][motif]['test_labels'] = pad_dataset(save['motif_dataset'][motif]['test_dataset'], save['motif_dataset'][motif]['test_labels'])
            dataset['motif_dataset'][motif]['train_dataset'], dataset['motif_dataset'][motif]['train_labels'] = pad_dataset(save['motif_dataset'][motif]['train_dataset'],save['motif_dataset'][motif]['train_labels'])
            dataset['motif_dataset'][motif]['valid_dataset'], dataset['motif_dataset'][motif]['valid_labels'] = pad_dataset(save['motif_dataset'][motif]['valid_dataset'], save['motif_dataset'][motif]['valid_labels'])

        test_results, motif_test_results= test(dataset, hyper_dict,i)

        print("Test accuracy: %.2f%%"%test_results[-1])
        for motif in HUMAN_MOTIF_VARIANTS:
            print('*%s* accuracy: %.1f%%' % (motif, motif_test_results[motif][-1]))

        test_accuracy_split.append(test_results[-1])
        for motif in HUMAN_MOTIF_VARIANTS:
            motif_test_accuracy_split[motif].append(motif_test_results[motif][-1])

    test_accuracy = np.mean(test_accuracy_split)
    motif_test_accuracy = {}
    for motif in HUMAN_MOTIF_VARIANTS:
        motif_test_accuracy[motif] = np.mean(motif_test_accuracy_split[motif])
    print('\n\n########################\nFinal result:')
    print('Test accuracy: %.1f%%' % (test_accuracy ))
    for motif in HUMAN_MOTIF_VARIANTS:
        print('*%s* accuracy: %.1f%%' % (motif, motif_test_accuracy[motif]))


    # enable_print()


if __name__ == '__main__':
    tf.app.run()