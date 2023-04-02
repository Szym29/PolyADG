#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
import sys, os
from decimal import Decimal
import hm_prep as hm
import ms_prep as ms
import random


############ Model Selection ############
HM_POS_PATH = 'data/human/omni_polyA_data/positive/'
HM_NEG_PATH = 'data/human/omni_polyA_data/negative/'
MS_POS_PATH = 'data/mouse/bl_mouse/positive/'
MS_NEG_PATH = 'data/mouse/bl_mouse/negative/'
RAT_NEG_PATH = 'data/rat/negative/'
RAT_POS_PATH = 'data/rat/positive/'
BATCH_SIZE = 256
PATCH_SIZE = 10
DEPTH = 16
NUM_HIDDEN = 128
SEQ_LEN = 206 + 2*PATCH_SIZE-2
NUM_CHANNELS = 4
NUM_LABELS = 2
NUM_EPOCHS = 100
NUM_FOLDS = 5
HYPER_DICT = None

############ **************** ############


FLAGS = tf.app.flags.FLAGS

def proportion(dataset,labels,fold):
    dataset_folds = np.array_split(dataset,10)
    labels_folds = np.array_split(labels,10)
    dataset = np.concatenate([dataset_folds[i] for i in fold['round']], axis=0)
    labels = np.concatenate([labels_folds[i] for i in fold['round']], axis=0)
def pad_dataset(dataset, labels):
    ''' Change dataset height to height + 2*DEPTH - 2'''
    new_dataset = np.ones([dataset.shape[0], dataset.shape[1]+2*PATCH_SIZE-2, dataset.shape[2], dataset.shape[3]], dtype = np.float32) * 0.25
    new_dataset[:, PATCH_SIZE-1:-(PATCH_SIZE-1), :, :] = dataset
    labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
    return new_dataset, labels

def pixel_level_shuffle(data):
    data = np.reshape(data,(data.shape[0],-1))
    shuffled_data = []
    for each in data:
        permutation = np.random.permutation(data.shape[1])
        each = each[permutation]
        shuffled_data.append(each)
    shuffled_data = np.array(shuffled_data)
    shuffled_data = np.reshape(shuffled_data, (-1, SEQ_LEN * NUM_CHANNELS))
    return shuffled_data
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def shuffle(dataset, labels, randomState=None):
    if randomState is None:
        permutation = np.random.permutation(labels.shape[0])
    else:
        permutation = randomState.permutation(labels.shape[0])

    shuffled_data = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_data, shuffled_labels

def gen_hyper_dict(hyper_dict=None):
    def rand_log(a, b):
        x = np.random.sample()
        return 10.0 ** ((np.log10(b) - np.log10(a)) * x + np.log10(a))

    def rand_sqrt(a, b):
        x = np.random.sample()
        return (b - a) * np.sqrt(x) + a

    if hyper_dict is None:
        hyper_dict = {


            'tf_learning_rate': 0.3,
            'tf_momentum': rand_sqrt(.90, .99),
            'tf_motif_init_weight': rand_log(1e-2, 1),
            'tf_fc_init_weight': rand_log(1e-2, 1),
            'tf_keep_prob': np.random.choice([.5, .75, 1.0]),
            'tf_ngroups': np.random.choice([2, 4, 8]),
            'tf_mlp_init_weight': rand_log(1e-2, 10),
            'tf_concat_init_weight': rand_log(1e-2, 1),
            'lambda': 0.06,
            'tf_keep_prob':0.2,
            

        }

    return hyper_dict


# Disable print
def block_print():
    sys.stdout = open(os.devnull, 'w')


def produce_labels(labels):
    labels = (np.arange(2) == labels[:,None]).astype(np.float32)
    return labels
# Restore print
def enable_print():
    sys.stdout = sys.__stdout__

def train(source_1, source_2, target,hyper_dict):
    graph = tf.Graph()
    with graph.as_default():

        # Load hyper-params
        tf_learning_rate = hyper_dict['tf_learning_rate']
        tf_momentum = hyper_dict['tf_momentum']
        tf_keep_prob = hyper_dict['tf_keep_prob']
        tf_mlp_init_weight = hyper_dict['tf_mlp_init_weight']
        tf_concat_init_weight = hyper_dict['tf_concat_init_weight']

        # Input data.

        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, SEQ_LEN, 1, NUM_CHANNELS))
        tr_shuffle =  tf.placeholder(
            tf.float32, shape=(BATCH_SIZE,  SEQ_LEN*NUM_CHANNELS))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))

        source_2_dataset = tf.constant(source_2['test_dataset'])
        source_2_shuffle =  tf.constant(pixel_level_shuffle(source_2['test_dataset']))
        source_2_label = tf.constant(source_2['test_labels'])

        
        source_1_dataset = tf.constant(source_1['test_dataset'])
        source_1_shuffle =  tf.constant(pixel_level_shuffle(source_1['test_dataset']))
        source_1_label = tf.constant(source_1['test_labels'])


        target_dataset = tf.constant(np.concatenate([target['train_dataset'], target['valid_dataset'], target['test_dataset']]))
        target_shuffle = tf.constant(pixel_level_shuffle(np.concatenate([target['train_dataset'], target['valid_dataset'], target['test_dataset']])))
        target_label = tf.constant(np.concatenate([target['train_labels'], target['valid_labels'], target['test_labels']]))


        tf_train_valid_dataset = tf.constant(np.concatenate([source_1['train_dataset'], source_2['train_dataset']]))
        tf_train_valid_shuffle =  tf.constant(pixel_level_shuffle(np.concatenate([source_1['train_dataset'], source_2['train_dataset']])))
        tf_train_valid_label = tf.constant(np.concatenate([source_1['train_labels'], source_2['train_labels']]))

        tf_valid_dataset = tf.constant(np.concatenate([source_1['valid_dataset'],source_2['valid_dataset']]))
        tf_valid_shuffle = tf.constant(pixel_level_shuffle(np.concatenate([source_1['valid_dataset'],source_2['valid_dataset']])))
        tf_valid_label = tf.constant(np.concatenate([source_1['valid_labels'],source_2['valid_labels']]))



        # Variables.
        conv_weights = tf.Variable(tf.truncated_normal(
          [PATCH_SIZE, 1, NUM_CHANNELS, DEPTH], stddev=1e-1))
        conv_weights_1 = tf.Variable(tf.truncated_normal(
          [PATCH_SIZE, 1, NUM_CHANNELS, DEPTH], stddev=1e-1))
        conv_biases = tf.Variable(tf.zeros([DEPTH]))
        conv_biases_1 = tf.Variable(tf.zeros([DEPTH]))
        layer1_weights = tf.Variable(tf.truncated_normal(
          [21*DEPTH, NUM_HIDDEN], stddev=1e-1))
        layer1_biases = tf.Variable(tf.constant(0.0, shape=[NUM_HIDDEN]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [NUM_HIDDEN, NUM_LABELS], stddev=1e-1))
        layer2_biases = tf.Variable(tf.constant(0.0, shape=[NUM_LABELS]))

        mlp_1_weights = tf.Variable(tf.truncated_normal(
            [ SEQ_LEN * NUM_CHANNELS, 128],stddev = tf_mlp_init_weight))
        mlp_out_weights = tf.Variable(tf.truncated_normal(
            [16,3],stddev = tf_mlp_init_weight))
        mlp_2_weights = tf.Variable(tf.truncated_normal(
            [512,128],stddev = tf_mlp_init_weight))
        concat_weights = tf.Variable(tf.truncated_normal(
            [128+128, NUM_LABELS],stddev = 1e-1))
        mlp_1_biases = tf.Variable(tf.constant(1.0, shape=[128]))
        mlp_2_biases = tf.Variable(tf.constant(0.0, shape=[128]))
        mlp_out_biases = tf.Variable(tf.constant(0.0, shape=[3]))
        concat_biases = tf.Variable(tf.constant(1.0, shape = [2]))
        conv1_w = tf.Variable(tf.truncated_normal(
          [3, 1, NUM_CHANNELS, DEPTH], stddev=1e-1))
        conv1_b = tf.Variable(tf.zeros([DEPTH]))

        # Store Variables
        weights = {}
        weights['conv_weights'] = conv_weights
        weights['conv_biases'] = conv_biases
        weights['layer1_weights'] = layer1_weights
        weights['layer1_biases'] = layer1_biases
        weights['layer2_weights'] = layer2_weights
        weights['layer2_biases'] = layer2_biases
        weights['mlp_1_weights'] = mlp_1_weights
        weights['mlp_1_biases'] = mlp_1_biases
        weights['mlp_2_weights'] = mlp_2_weights
        weights['mlp_2_biases'] = mlp_2_biases
        weights['concat_weights'] = concat_weights
        weights['concat_biases'] = concat_biases
        weights['back_biases'] = mlp_out_biases   
        weights['back_weights'] = mlp_out_weights

        # Model.

        def model(data, shuffle, label, drop=True):
            
        # MLP
            Hex = True
            mlp_1 = tf.nn.relu(tf.matmul(shuffle, mlp_1_weights) + mlp_1_biases)

        # CNN
            conv = tf.nn.conv2d(data, conv_weights, [1, 1, 1, 1], padding='VALID')

            hidden = tf.nn.relu(conv)
            hidden = tf.nn.max_pool(hidden, [1, 10, 1, 1], [1, 10, 1, 1], padding='VALID')

            shape = hidden.get_shape().as_list()
            motif_score = tf.reshape(hidden, [shape[0], shape[1] * DEPTH])

            hidden_nodes = tf.nn.dropout(tf.nn.relu(tf.matmul(motif_score, layer1_weights) + layer1_biases),
                                         tf_keep_prob)
            concat_loss = tf.concat([hidden_nodes, mlp_1], 1)

            pad = tf.zeros_like(mlp_1, tf.float32)
            concat_pred = tf.concat([hidden_nodes, pad], 1)

            pad2 = tf.zeros_like(hidden_nodes, tf.float32)
            concat_H = tf.concat([pad2, mlp_1], 1)
            model_loss = tf.matmul(concat_loss, concat_weights) + concat_biases 
            model_pred = tf.matmul(concat_pred, concat_weights) + concat_biases
            model_H = tf.matmul(concat_H, concat_weights) + concat_biases

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                                              logits=model_loss))
            if Hex:
                model_loss = tf.nn.l2_normalize(model_loss, 0)
                model_H = tf.nn.l2_normalize(model_H, 0)
                model_loss = model_loss - \
                         tf.matmul(tf.matmul(
                             tf.matmul(model_H, tf.matrix_inverse(tf.matmul(model_H, model_H, transpose_a=True))),
                             model_H, transpose_b=True), model_loss)

                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, 
                                                                              logits=model_loss))
                
            return loss, model_pred

    # Training computation.
        print('???')
        print(tr_shuffle.shape)
        loss, _ = model(tf_train_dataset, tr_shuffle, tf_train_labels, drop=True)
    # Optimizer.
        global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
        stepOp = tf.assign_add(global_step, 1).op
        learning_rate = tf.train.exponential_decay(tf_learning_rate, global_step, 3000, 0.96)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
        motif_train_prediction = {}
        print(tf_train_valid_shuffle.shape)
        train_loss, train_valid = model(tf_train_valid_dataset,tf_train_valid_shuffle, tf_train_valid_label, drop=True)
        train_prediction = tf.nn.softmax(train_valid)

        valid_loss, validation = model(tf_valid_dataset,tf_valid_shuffle, tf_valid_label, drop=True)
        valid_prediction = tf.nn.softmax(validation)       

        _,source_1_out = model(source_1_dataset,source_1_shuffle, source_1_label,drop = True)
        source_1_prediction = tf.nn.softmax(source_1_out)

        _, source_2_out = model(source_2_dataset, source_2_shuffle,source_2_label, drop=True)
        source_2_prediction = tf.nn.softmax(source_2_out)


        _, target_out = model(target_dataset,target_shuffle, target_label, drop=True)
        target_prediction = tf.nn.softmax(target_out)


# Kick off training
    train_resuts = []
    valid_results = []
    source_1_results = []
    source_2_results = []
    target_results = []


    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        train_dataset = np.concatenate([source_1['train_dataset'],source_2['train_dataset']])
        train_labels =  np.concatenate([source_1['train_labels'],source_2['train_labels']])
        np.random.seed()
        print('Initialized')
        print('Training accuracy at the beginning: %.1f%%' % accuracy(train_prediction.eval(), np.concatenate([source_1['train_labels'], source_2['train_labels']])))
        print('Validation accuracy at the beginning: %.1f%%' % accuracy(valid_prediction.eval(), np.concatenate([source_1['valid_labels'],source_2['valid_labels']])))
        for epoch in range(NUM_EPOCHS):
            permutation = np.random.permutation(train_labels.shape[0])
            shuffled_dataset = train_dataset[permutation, :, :]
            shuffled_labels = train_labels[permutation, :]
            # shuffled_domains = domain[permutation, :]
            for step in range(shuffled_labels.shape[0] // BATCH_SIZE):
                offset = step * BATCH_SIZE
                batch_data = shuffled_dataset[offset:(offset + BATCH_SIZE), :, :, :]
                batch_labels = shuffled_labels[offset:(offset + BATCH_SIZE), :]
                batch_shuffle = pixel_level_shuffle(batch_data)
                feed_dict = {tf_train_dataset: batch_data,tr_shuffle :batch_shuffle, tf_train_labels: batch_labels}
                _, l = session.run(
                    [optimizer, loss], feed_dict=feed_dict)
                session.run(stepOp)
            train_resuts.append(accuracy(train_prediction.eval(), np.concatenate([source_1['train_labels'], source_2['train_labels']])))

            valid_pred = valid_prediction.eval()
            print('validation loss', valid_loss.eval())
            # valid_losses.append(valid_loss.eval())
            valid_results.append(accuracy(valid_pred,  np.concatenate([source_1['valid_labels'], source_2['valid_labels']])))
            source_1_pred = source_1_prediction.eval()
            source_1_results.append(accuracy(source_1_pred,source_1['test_labels']))

            source_2_pred = source_2_prediction.eval()
            source_2_results.append(accuracy(source_2_pred, source_2['test_labels']))

            target_pred = target_prediction.eval()
            target_results.append(accuracy(target_pred, np.concatenate([target['train_labels'], target['valid_labels'], target['test_labels']])))


            print('Training accuracy at epoch %d: %.1f%%' % (epoch, train_resuts[-1]))
            print('Validation accuracy: %.1f%%' % valid_results[-1])
            print('target accuracy:%.1f%%'%target_results[-1])

        # Early stopping based on validation results
            if epoch > 10 and valid_results[-11] > max(valid_results[-10:]):
                train_resuts = train_resuts[:-10]
                valid_results = valid_results[:-10]
                source_1_results = source_1_results[:-10]
                source_2_results = source_2_results[:-10]
                target_results = target_results[:-10]

                return train_resuts, valid_results, source_1_results, source_2_results, target_results 
    return train_resuts, valid_results, source_1_results, source_2_results, target_results


def main(_):

    for rounds in range(0,10):
        print('rounds:%d'%rounds)
        hyper_dict = gen_hyper_dict(HYPER_DICT)
        source_1_pos_data, source_1_pos_labels, source_1_neg_data, source_1_neg_labels = hm.produce_dataset(NUM_FOLDS, HM_POS_PATH,HM_NEG_PATH)
        source_2_pos_data, source_2_pos_labels, source_2_neg_data, source_2_neg_labels = ms.produce_dataset(NUM_FOLDS,MS_POS_PATH,
                                                                                                   MS_NEG_PATH)
        target_pos_data, target_pos_labels, target_neg_data, target_neg_labels = ms.produce_dataset(NUM_FOLDS,RAT_POS_PATH,
                                                                                                   RAT_NEG_PATH)

    # Cross validate
        train_accuracy_split = []
        valid_accuracy_split = []
        source_2_accuracy_split = []
        target_accuracy_split = []
        source_1_accuracy_split = []

        folds = {'round':[k for k in range(rounds+1)]}
        for i in range(NUM_FOLDS):
            split =  {
            'train': [(i + j) % NUM_FOLDS for j in range(NUM_FOLDS-2)], 
            'valid': [(i + NUM_FOLDS-2) % NUM_FOLDS], 
            'test': [(i + NUM_FOLDS-1) % NUM_FOLDS]
            }
            #proportion_60 = {'round':[k for k in range(0,6)]}
            #proportion_100 = {'round':[k for k in range(0,10)]}
            source_1_buffer = hm.data_split(source_1_pos_data, source_1_pos_labels, source_1_neg_data, source_1_neg_labels, NUM_FOLDS, split,folds)
            source_2_buffer = ms.data_split(source_2_pos_data, source_2_pos_labels, source_2_neg_data, source_2_neg_labels, NUM_FOLDS, split,folds)
            target_buffer = ms.data_split(target_pos_data, target_pos_labels, target_neg_data, target_neg_labels, NUM_FOLDS, split, folds)

            source_1 = {}
            source_1['train_dataset'], source_1['train_labels'] = pad_dataset(source_1_buffer['train_dataset'], source_1_buffer['train_labels'])
            source_1['valid_dataset'], source_1['valid_labels'] = pad_dataset(source_1_buffer['valid_dataset'], source_1_buffer['valid_labels'])
            source_1['test_dataset'], source_1['test_labels'] = pad_dataset(source_1_buffer['test_dataset'], source_1_buffer['test_labels'])

            source_2 = {}
            source_2['train_dataset'], source_2['train_labels'] = pad_dataset(source_2_buffer['train_dataset'], source_2_buffer['train_labels'])
            source_2['valid_dataset'], source_2['valid_labels'] = pad_dataset(source_2_buffer['valid_dataset'], source_2_buffer['valid_labels'])
            source_2['test_dataset'], source_2['test_labels'] = pad_dataset(source_2_buffer['test_dataset'], source_2_buffer['test_labels'])

            target = {}
            target['train_dataset'], target['train_labels'] = pad_dataset(target_buffer['train_dataset'], target_buffer['train_labels'])
            target['valid_dataset'], target['valid_labels'] = pad_dataset(target_buffer['valid_dataset'], target_buffer['valid_labels'])
            target['test_dataset'], target['test_labels'] = pad_dataset(target_buffer['test_dataset'], target_buffer['test_labels'])


            train_resuts, valid_results, source_1_results, source_2_results, target_results = train(source_1, source_2, target,hyper_dict)
            
            print("\nbest valid epoch: %d" % (len(train_resuts) - 1))
            print("Training accuracy: %.2f%%" % train_resuts[-1])
            print("Validation accuracy: %.2f%%" % valid_results[-1])
            print("Source_1 training accuracy:%.2f%%"%source_1_results[-1])
            print("Source_2 training accuracy: %.2f%%" % source_1_results[-1])
            print("Traget accuracy: %.2f%%" % target_results[-1])



            train_accuracy_split.append(train_resuts[-1])
            valid_accuracy_split.append(valid_results[-1])
            source_1_accuracy_split.append(source_1_results[-1])
            source_2_accuracy_split.append(source_2_results[-1])
            target_accuracy_split.append(target_results[-1])



        train_accuracy = np.mean(train_accuracy_split)
        valid_accuracy = np.mean(valid_accuracy_split)
        source_1_accuracy = np.mean(source_1_accuracy_split)
        source_2_accuracy = np.mean(source_2_accuracy_split)
        target_accuracy = np.mean(target_accuracy_split)

        print('\n\n########################\nFinal result:')
        print('Training set accuracy: %.1f%%' % (train_accuracy))
        print('Validation set accuracy: %.1f%%' % (valid_accuracy))
        print("Source_1 accuracy:%.1f%%"% (source_1_accuracy))
        print('Srouce_2 accuracy: %.1f%%' % (source_2_accuracy))
        print('Target accuracy: %.1f%%' % (target_accuracy))



if __name__ == '__main__':
    tf.app.run()









