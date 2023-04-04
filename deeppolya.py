#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf
from six.moves import cPickle as pickle
import sys, os
from decimal import Decimal
import hm_prep as hm
import ms_prep as ms
import random
import argparse

############ Model Selection ############
HM_POS_PATH = 'data/human/omni_polyA_data/positive/'
HM_NEG_PATH = 'data/human/omni_polyA_data/negative/'
MS1_POS_PATH = 'data/mouse/bl_mouse/positive/'
MS1_NEG_PATH = 'data/mouse/bl_mouse/negative/'
RAT_NEG_PATH = 'data/rat/negative/'
RAT_POS_PATH = 'data/rat/positive/'
BATCH_SIZE = 128
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


            'tf_learning_rate': 0.001,
            'tf_motif_init_weight': rand_log(1e-2, 1),
            'tf_fc_init_weight': rand_log(1e-2, 1),
            'tf_keep_prob': np.random.choice([.5, .75, 1.0]),
            'tf_ngroups': np.random.choice([2, 4, 8]),
            'tf_mlp_init_weight': rand_log(1e-2, 10),
            'tf_concat_init_weight': rand_log(1e-2, 1),
            'tf_keep_prob':0.2,
            

        }
    return hyper_dict


# Disable print
def block_print():
    sys.stdout = open(os.devnull, 'w')

def testScores(X,domain,label):
    pl = []
    ll = []
    for i in range(domain.shape[0]):

        print (i)

        nb = GaussianNB()
        ps = cross_val_score(nb, X, p, cv=3)
        ls = cross_val_score(nb, X, l, cv=3)

        pl.append(np.mean(ps))
        ll.append(np.mean(ls))

    return np.array(pl), np.array(ll)


def calculateScores(X,domain,label):


    domain, label = testScores(X,domain,label)

    print( np.mean(domain), np.std(domain), np.mean(label), np.std(label))
    plt.plot(domain, ls='-', color='r')
    plt.plot(label, ls='-.', color='b')

    plt.show()

def produce_labels(labels):
    labels = (np.arange(2) == labels[:,None]).astype(np.float32)
    return labels
# Restore print
def enable_print():
    sys.stdout = sys.__stdout__

def train(dataset, d2,d3,d4,hyper_dict):
    graph = tf.Graph()
    with graph.as_default():

        # Load hyper-params
        tf_learning_rate = hyper_dict['tf_learning_rate']
        #tf_momentum = hyper_dict['tf_momentum']
        tf_keep_prob = hyper_dict['tf_keep_prob']
       # tf_ngroups = hyper_dict['tf_ngroups']
        tf_mlp_init_weight = hyper_dict['tf_mlp_init_weight']
        tf_concat_init_weight = hyper_dict['tf_concat_init_weight']
        #lamda = hyper_dict['lambda']
        # Input data.
        l = tf.placeholder(tf.float32, [])
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, SEQ_LEN, 1, NUM_CHANNELS))
        tr_shuffle =  tf.placeholder(
            tf.float32, shape=(BATCH_SIZE,  NUM_CHANNELS))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
        
        tf_d2_dataset = tf.constant(
            d2['test_dataset'])
        tf_d2_shuffle =  tf.constant(pixel_level_shuffle(d2['test_dataset']))
        tf_d2_label = tf.constant(d2['test_labels'])


        
        tf_d1_dataset = tf.constant(dataset['test_dataset'])
        tf_d1_shuffle =  tf.constant(pixel_level_shuffle(dataset['test_dataset']))
        tf_d1_label = tf.constant(dataset['test_labels'])
       
           

        tf_d3_dataset = tf.constant(np.concatenate([d3['train_dataset'],d3['valid_dataset'],
           d3['test_dataset']]))
        tf_d3_label = tf.constant(np.concatenate([d3['train_labels'],d3['valid_labels'],
            d3['test_labels']]))
        
        tf_d4_dataset = tf.constant(np.concatenate([d4['train_dataset'],d4['valid_dataset'],d4['test_dataset']]))
        tf_d4_shuffle = tf.constant(pixel_level_shuffle(np.concatenate([d4['train_dataset'],d4['valid_dataset'],d4['test_dataset']])))
        tf_d4_label = tf.constant(np.concatenate([d4['train_labels'],d4['valid_labels'],d4['test_labels']]))

        tf_train_valid_dataset = tf.constant(np.concatenate([dataset['train_dataset'], d2['train_dataset']]))
        tf_train_valid_label = tf.constant(np.concatenate([dataset['train_labels'], d2['train_labels']]))

       
        tf_valid_dataset = tf.constant(np.concatenate([dataset['valid_dataset'],d2['valid_dataset']]))
        tf_valid_label = tf.constant(np.concatenate([dataset['valid_labels'],d2['valid_labels']]))


        # Variables.
        conv1_weights = tf.Variable(tf.truncated_normal(
          [8, 1, NUM_CHANNELS, DEPTH], stddev=1e-1))#4,8,16
        conv2_weights = tf.Variable(tf.truncated_normal(
          [6, 1, 16, 64], stddev=1e-1))
        conv2_biases = tf.Variable(tf.zeros([64]))
        fc1_weights = tf.Variable(tf.truncated_normal(
            [6784,64],stddev = tf_mlp_init_weight))
        fc2_weights = tf.Variable(tf.truncated_normal(
            [64,2],stddev = tf_mlp_init_weight))
        fc1_biases = tf.Variable(tf.constant(0.0, shape=[64]))
        fc2_biases = tf.Variable(tf.constant(0.0, shape=[2]))
        

        # Model.

        def model(data, label, drop=True):
          
          #conv1
            conv1 = tf.nn.relu(tf.nn.conv2d(data, conv1_weights,[1, 1, 1, 1], padding='VALID'))
        #dropout1
            dropout1 = tf.nn.dropout(conv1, 0.6)
        #conv2 
            conv2 = tf.nn.relu(tf.nn.conv2d(dropout1, conv2_weights, [1, 1, 1, 1], padding='VALID'))
        #maxpooling
            
            max_pool = tf.nn.max_pool(conv2,[1,2,1,1],[1,2,1,1], padding='VALID')
        #dropout2 
            dropout2 = tf.nn.dropout(max_pool, 0.6)
        #fc1
            dropout2 = tf.reshape(dropout2,[dropout2.shape[0], -1])
            fc1 = tf.nn.dropout(tf.matmul(dropout2, fc1_weights)+fc1_biases, 0.6)

        #fc2 
            fc2 = tf.matmul(fc1, fc2_weights)+fc2_biases
        #output
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits=fc2))
                
            return loss, fc2

    # Training computation.
        loss, _ = model(tf_train_dataset, tf_train_labels, drop=True)
    # Optimizer.
        #global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
        #stepOp = tf.assign_add(global_step, 1).op
        #learning_rate = tf.train.exponential_decay(tf_learning_rate, global_step, 3000, 0.96)
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)
    # Predictions for the training, validation, and test data.
        motif_train_prediction = {}

        train_loss, train_valid = model(tf_train_valid_dataset, tf_train_valid_label, drop=True)
        train_prediction = tf.nn.softmax(train_valid)
        _,d1_out = model(tf_d1_dataset, tf_d1_label,drop = True)
        d1_prediction = tf.nn.softmax(d1_out)


        valid_loss, validation = model(tf_valid_dataset, tf_valid_label, drop=True)
        valid_prediction = tf.nn.softmax(validation)
        _, test1 = model(tf_d2_dataset,tf_d2_label, drop=True)
        test1_prediction = tf.nn.softmax(test1)


        _, test2 = model(tf_d3_dataset, tf_d3_label, drop=True)
        test2_prediction = tf.nn.softmax(test2)
            
        _, d4_out = model(tf_d4_dataset, tf_d4_label, drop=True)
        d4_prediction = tf.nn.softmax(d4_out)

# Kick off training
    valid_losses = []
    train_resuts = []
    valid_results = []
    d2_results = []
    d3_results = []
    d4_results = []
    d1_results = []
    save_weights = []

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        train_dataset = np.concatenate([dataset['train_dataset'],d2['train_dataset']])
        train_labels =  np.concatenate([dataset['train_labels'],d2['train_labels']])
        np.random.seed()
        print('Initialized')
        print('Training accuracy at the beginning: %.1f%%' % accuracy(train_prediction.eval(), np.concatenate([dataset['train_labels'], d2['train_labels']])))
        print('Validation accuracy at the beginning: %.1f%%' % accuracy(valid_prediction.eval(), np.concatenate([dataset['valid_labels'],d2['valid_labels']])))
        for epoch in range(NUM_EPOCHS):
            Reps = None
            Label = None
            Pattern = None
            
            permutation = np.random.permutation(train_labels.shape[0])
            shuffled_dataset = train_dataset[permutation, :, :]
            shuffled_labels = train_labels[permutation, :]
            for step in range(shuffled_labels.shape[0] // BATCH_SIZE):
                offset = step * BATCH_SIZE
                batch_data = shuffled_dataset[offset:(offset + BATCH_SIZE), :, :, :]
                batch_labels = shuffled_labels[offset:(offset + BATCH_SIZE), :]
                batch_shuffle = pixel_level_shuffle(batch_data)
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                _, l = session.run(
                    [optimizer, loss], feed_dict=feed_dict)
                #session.run(stepOp)
            train_resuts.append(accuracy(train_prediction.eval(), np.concatenate([dataset['train_labels'], d2['train_labels']])))

            valid_pred = valid_prediction.eval()
            #print('validation loss', valid_loss.eval())
            valid_losses.append(valid_loss.eval())
            valid_results.append(accuracy(valid_pred,  np.concatenate([dataset['valid_labels'],d2['valid_labels']])))
            d1_pred = d1_prediction.eval()
            d1_results.append(accuracy(d1_pred,dataset['test_labels']))
            d2_pred = test1_prediction.eval()
            d2_results.append(accuracy(d2_pred, d2['test_labels']))

            d3_pred = test2_prediction.eval()
            d3_results.append(accuracy(d3_pred, np.concatenate([d3['train_labels'],d3['valid_labels'],
            d3['test_labels']])))
            
            d4_pred = d4_prediction.eval()
            d4_results.append(accuracy(d4_pred, np.concatenate([d4['train_labels'],d4['valid_labels'],
            d4['test_labels']])))


            #print('Training accuracy at epoch %d: %.1f%%' % (epoch, train_resuts[-1]))
            #print('Validation accuracy: %.1f%%' % valid_results[-1])
            #print('target 1 accuracy:%.1f%%'%d3_results[-1])
            #print('target 2 accuracy:%.1f%%'%d4_results[-1])
        # Early stopping based on validation results
            if epoch > 10 and valid_results[-11] > max(valid_results[-10:]):
                train_resuts = train_resuts[:-10]
                valid_results = valid_results[:-10]
                d1_results = d1_results[:-10]
                d2_results = d2_results[:-10]
                d3_results = d3_results[:-10]
                d4_results = d4_results[:-10]

                return train_resuts, valid_results, d2_results, d3_results,d4_results, valid_losses,d1_results


    return train_resuts, valid_results, d2_results, d3_results,d4_results,valid_losses,d1_results


def main():
    for rounds in range(0,10):
        print('rounds:%d'%rounds)
        hyper_dict = gen_hyper_dict(HYPER_DICT)
        d1_pos_data, d1_pos_labels, d1_neg_data, d1_neg_labels = hm.produce_dataset(NUM_FOLDS, D1_POS_PATH,D1_NEG_PATH)
        d2_pos_data, d2_pos_labels, d2_neg_data, d2_neg_labels = ms.produce_dataset(NUM_FOLDS,D2_POS_PATH,D2_NEG_PATH)
        d3_pos_data, d3_pos_labels, d3_neg_data, d3_neg_labels = ms.produce_dataset(NUM_FOLDS,D3_POS_PATH,D3_NEG_PATH)
        d4_pos_data, d4_pos_labels, d4_neg_data, d4_neg_labels = hm.produce_dataset(NUM_FOLDS,D4_POS_PATH,D4_NEG_PATH)
        
    # Cross validate
        train_accuracy_split = []
        valid_accuracy_split = []
        d2_accuracy_split = []
        d3_accuracy_split = []
        d4_accuracy_split = []
        d1_accuracy_split = []

        folds = {'round':[k for k in range(rounds+1)]}
        for i in range(NUM_FOLDS):
            split =  {
            'train': [(i + j) % NUM_FOLDS for j in range(NUM_FOLDS-2)], 
            'valid': [(i + NUM_FOLDS-2) % NUM_FOLDS], 
            'test': [(i + NUM_FOLDS-1) % NUM_FOLDS]
            }

            print(split['train'])
            print(split['valid'])
            print(split['test'])
            split1= {
            'train': [(i + j) % NUM_FOLDS for j in range(NUM_FOLDS-2)], 
            'valid': [(i + NUM_FOLDS-2) % NUM_FOLDS], 
            'test': [(i + NUM_FOLDS-1) % NUM_FOLDS]
            }
            fold60 = {'round':[k for k in range(0,6)]}
            fold100 = {'round':[k for k in range(0,10)]}
            
            
            d1_temp = hm.data_split(d1_pos_data, d1_pos_labels, d1_neg_data, d1_neg_labels, NUM_FOLDS, split,folds)
            d2_temp = hm.data_split(d2_pos_data, d2_pos_labels, d2_neg_data, d2_neg_labels, NUM_FOLDS, split,folds)
            d3_temp = ms.data_split(d3_pos_data, d3_pos_labels, d3_neg_data, d3_neg_labels, NUM_FOLDS,
                                          split,fold100)
            d4_temp = hm.data_split(d4_pos_data, d4_pos_labels, d4_neg_data, d4_neg_labels, NUM_FOLDS,
                                          split1,fold100)
            train_split = {}
            valid_split = {}

            d1 = {}
            d1['train_dataset'], d1['train_labels'] = pad_dataset(d1_temp['train_dataset'], d1_temp['train_labels'])
            d1['valid_dataset'], d1['valid_labels'] = pad_dataset(d1_temp['valid_dataset'], d1_temp['valid_labels'])
            d1['test_dataset'], d1['test_labels'] = pad_dataset(d1_temp['test_dataset'], d1_temp['test_labels'])
            
            d2 = {}
            d2['train_dataset'], d2['train_labels'] = pad_dataset(d2_temp['train_dataset'], d2_temp['train_labels'])
            d2['valid_dataset'], d2['valid_labels'] = pad_dataset(d2_temp['valid_dataset'], d2_temp['valid_labels'])
            d2['test_dataset'], d2['test_labels'] = pad_dataset(d2_temp['test_dataset'], d2_temp['test_labels'])

            d3 = {}
            d3['train_dataset'], d3['train_labels'] = pad_dataset(d3_temp['train_dataset'], d3_temp['train_labels'])
            d3['valid_dataset'], d3['valid_labels'] = pad_dataset(d3_temp['valid_dataset'], d3_temp['valid_labels'])
            d3['test_dataset'], d3['test_labels'] = pad_dataset(d3_temp['test_dataset'], d3_temp['test_labels'])

            d4 = {}
            d4['train_dataset'], d4['train_labels'] = pad_dataset(d4_temp['train_dataset'], d4_temp['train_labels'])
            d4['valid_dataset'], d4['valid_labels'] = pad_dataset(d4_temp['valid_dataset'], d4_temp['valid_labels'])
            d4['test_dataset'], d4['test_labels'] = pad_dataset(d4_temp['test_dataset'], d4_temp['test_labels'])

            train_resuts, valid_results, d2_results, d3_results, d4_results, valid_loss,d1_results = train(d1, d2, d3,d4,hyper_dict)
            
            print(valid_loss)
            print("\nbest valid epoch: %d" % (len(train_resuts) - 1))
            print("Training accuracy: %.2f%%" % train_resuts[-1])
            print("Tr_2 accuracy: %.2f%%" % d2_results[-1])
            print("Traget 1 accuracy: %.2f%%" % d3_results[-1])
            print("Traget 2 accuracy: %.2f%%" % d4_results[-1])
            print("Validation accuracy: %.2f%%" % valid_results[-1])
            print("Tr_1 accuracy:%.2f%%"%d1_results[-1])

            train_accuracy_split.append(train_resuts[-1])
            valid_accuracy_split.append(valid_results[-1])
            d2_accuracy_split.append(d2_results[-1])
            d3_accuracy_split.append(d3_results[-1])
            d4_accuracy_split.append(d4_results[-1])
            d1_accuracy_split.append(d1_results[-1])


        train_accuracy = np.mean(train_accuracy_split)
        valid_accuracy = np.mean(valid_accuracy_split)
        d2_accuracy = np.mean(d2_accuracy_split)
        d3_accuracy = np.mean(d3_accuracy_split)
        d4_accuracy = np.mean(d4_accuracy_split)
        d1_accuracy = np.mean(d1_accuracy_split)
        with open('./results/deeppolya.txt','a+') as f:
            f.write('source are %s+%s, target are %s and %s \n'% (DOMAIN1,DOMAIN2,DOMAIN3,DOMAIN4))
            f.write('rounds is %d\n'%(rounds+1))
            f.write('%s accuracy: %.1f%% \n' %(DOMAIN3,d3_accuracy))
            f.write('%s accuracy: %.1f%% \n\n' %(DOMAIN4,d4_accuracy))


parser = argparse.ArgumentParser()
parser.add_argument("--d1", "-1", type=int, default=1)
parser.add_argument("--d2", "-2", type=int, default=2)
parser.add_argument("--d3", "-3", type=int, default=3)
parser.add_argument("--d4", "-4", type=int, default=4)
args = parser.parse_args()

if args.d1== 1:
    DOMAIN1='human'
    D1_POS_PATH ='data/human/omni_polyA_data/positive/'
    D1_NEG_PATH ='data/human/omni_polyA_data/negative/'
if args.d1== 2:
    DOMAIN1='mouse'
    D1_POS_PATH ='data/mouse/bl_mouse/positive/'
    D1_NEG_PATH ='data/mouse/bl_mouse/negative/'
if args.d1== 3:
    DOMAIN1='rat'
    D1_NEG_PATH = 'data/rat/negative/'
    D1_POS_PATH = 'data/rat/positive/'
if args.d1== 4:
    DOMAIN1='bovine'
    D1_NEG_PATH = 'data/bovine/negative/'
    D1_POS_PATH = 'data/bovine/positive/'
if args.d2== 1:
    DOMAIN2='human'
    D2_POS_PATH ='data/human/omni_polyA_data/positive/'
    D2_NEG_PATH ='data/human/omni_polyA_data/negative/'
if args.d2== 2:
    DOMAIN2='mouse'
    D2_POS_PATH ='data/mouse/bl_mouse/positive/'
    D2_NEG_PATH ='data/mouse/bl_mouse/negative/'
if args.d2== 3:
    DOMAIN2='rat'
    D2_NEG_PATH = 'data/rat/negative/'
    D2_POS_PATH = 'data/rat/positive/'
if args.d2== 4:
    DOMAIN2='bovine'
    D2_NEG_PATH = 'data/bovine/negative/'
    D2_POS_PATH = 'data/bovine/positive/'
if args.d3== 1:
    DOMAIN3='human'
    D3_POS_PATH ='data/human/omni_polyA_data/positive/'
    D3_NEG_PATH ='data/human/omni_polyA_data/negative/'
if args.d3== 2:
    DOMAIN3='mouse'
    D3_POS_PATH ='data/mouse/bl_mouse/positive/'
    D3_NEG_PATH ='data/mouse/bl_mouse/negative/'
if args.d3== 3:
    DOMAIN3='rat'
    D3_NEG_PATH = 'data/rat/negative/'
    D3_POS_PATH = 'data/rat/positive/'
if args.d3== 4:
    DOMAIN3='bovine'
    D3_NEG_PATH = 'data/bovine/negative/'
    D3_POS_PATH = 'data/bovine/positive/'
if args.d4== 1:
    DOMAIN4='human'
    D4_POS_PATH ='data/human/omni_polyA_data/positive/'
    D4_NEG_PATH ='data/human/omni_polyA_data/negative/'
if args.d4== 2:
    DOMAIN4='mouse'
    D4_POS_PATH ='data/mouse/bl_mouse/positive/'
    D4_NEG_PATH ='data/mouse/bl_mouse/negative/'
if args.d4== 3:
    DOMAIN4='rat'
    D4_NEG_PATH = 'data/rat/negative/'
    D4_POS_PATH = 'data/rat/positive/'
if args.d4== 4:
    DOMAIN4='bovine'
    D4_NEG_PATH = 'data/bovine/negative/'
    D4_POS_PATH = 'data/bovine/positive/'
main()










