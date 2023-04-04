#!/usr/bin/env python3
from SANPolyA import *
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import sys, os
from decimal import Decimal
import hm_prep as hm
import ms_prep as ms
import random
import argparse
from keras.callbacks import EarlyStopping
############ Model Selection ############
D1_POS_PATH = 'data/human/omni_polyA_data/positive/'
D1_NEG_PATH = 'data/human/omni_polyA_data/negative/'
D2_POS_PATH = 'data/mouse/bl_mouse/positive/'
D2_NEG_PATH = 'data/mouse/bl_mouse/negative/'
D3_NEG_PATH = 'data/rat/negative/'
D3_POS_PATH = 'data/rat/positive/'
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
    return dataset, labels
    
def pad_dataset(dataset, labels):
    ''' Change dataset height to height + 2*DEPTH - 2'''
    #new_dataset = np.ones([dataset.shape[0], dataset.shape[1]+2*PATCH_SIZE-2, dataset.shape[2], dataset.shape[3]], dtype = np.float32) * 0.25
    #new_dataset[:, PATCH_SIZE-1:-(PATCH_SIZE-1), :, :] = dataset
    #labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
    return dataset, labels

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




def main():
    for rounds in range(0,10):
        acc = 0
        acc1=0
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
            
            ##pos neg proportion##
            #print(d1_pos_data.shape,d1_pos_labels.shape)
            #d1_pos_data, d1_pos_labels = proportion(d1_pos_data_round, d1_pos_labels_round,folds)
            #d1_neg_data, d1_neg_labels = proportion(d1_neg_data_round, d1_neg_labels_round,folds)
            #d3_pos_data, d3_pos_labels = proportion(d3_pos_data_round, d3_pos_labels_round,folds)
            #d3_neg_data, d3_neg_labels = proportion(d3_neg_data_round, d3_neg_labels_round,folds)
            #d2_pos_data, d2_pos_labels = proportion(d2_pos_data_round, d2_pos_labels_round,folds)
            #d2_neg_data, d2_neg_labels = proportion(d2_neg_data_round, d2_neg_labels_round,folds)
            d1_temp = hm.data_split(d1_pos_data, d1_pos_labels, d1_neg_data, d1_neg_labels, NUM_FOLDS, split,folds)
            
            d2_temp = ms.data_split(d2_pos_data, d2_pos_labels, d2_neg_data, d2_neg_labels, NUM_FOLDS,
                                          split1,folds)
            d3_temp = ms.data_split(d3_pos_data, d3_pos_labels, d3_neg_data, d3_neg_labels, NUM_FOLDS,
                                          split1,fold100)
            d4_temp = hm.data_split(d4_pos_data, d4_pos_labels, d4_neg_data, d4_neg_labels, NUM_FOLDS,
                                          split1,fold100)
            train_split = {}
            valid_split = {}
            d2 = {}
            d2['train_dataset'], d2['train_labels'] = pad_dataset(d2_temp['train_dataset'], d2_temp['train_labels'])
            d2['valid_dataset'], d2['valid_labels'] = pad_dataset(d2_temp['valid_dataset'], d2_temp['valid_labels'])
            d2['test_dataset'], d2['test_labels'] = pad_dataset(d2_temp['test_dataset'], d2_temp['test_labels'])

            d3 = {}
            d3['train_dataset'], d3['train_labels'] = pad_dataset(d3_temp['train_dataset'], d3_temp['train_labels'])
            d3['valid_dataset'], d3['valid_labels'] = pad_dataset(d3_temp['valid_dataset'], d3_temp['valid_labels'])
            d3['test_dataset'], d3['test_labels'] = pad_dataset(d3_temp['test_dataset'], d3_temp['test_labels'])

            d1 = {}
            d1['train_dataset'], d1['train_labels'] = pad_dataset(d1_temp['train_dataset'], d1_temp['train_labels'])
            d1['valid_dataset'], d1['valid_labels'] = pad_dataset(d1_temp['valid_dataset'], d1_temp['valid_labels'])
            d1['test_dataset'], d1['test_labels'] = pad_dataset(d1_temp['test_dataset'], d1_temp['test_labels'])
            
            d4 = {}
            d4['train_dataset'], d4['train_labels'] = pad_dataset(d4_temp['train_dataset'], d4_temp['train_labels'])
            d4['valid_dataset'], d4['valid_labels'] = pad_dataset(d4_temp['valid_dataset'], d4_temp['valid_labels'])
            d4['test_dataset'], d4['test_labels'] = pad_dataset(d4_temp['test_dataset'], d4_temp['test_labels'])
            ##
            x_train = np.concatenate([d1['train_dataset'],d2['train_dataset']])
            y_train = np.concatenate([d1['train_labels'],d2['train_labels']])
            
            x_test = np.concatenate([d1['test_dataset'],d2['test_dataset']])
            y_test = np.concatenate([d1['test_labels'],d2['test_labels']])
            
            x_valid = np.concatenate([d1['valid_dataset'],d2['valid_dataset']])
            y_valid = np.concatenate([d1['valid_labels'],d2['valid_labels']])
            
            x_target = np.concatenate([d3['train_dataset'],d3['valid_dataset'],d3['test_dataset']])
            y_target = np.concatenate([d3['train_labels'],d3['valid_labels'],d3['test_labels']])
            
            x1_target = np.concatenate([d4['train_dataset'],d4['valid_dataset'],d4['test_dataset']])
            y1_target = np.concatenate([d4['train_labels'],d4['valid_labels'],d4['test_labels']])
            #print(x_train.shape)
            #exit()
            x_train = x_train.reshape([-1,206,4])
            x_valid = x_valid.reshape([-1,206,4])
            x_target = x_target.reshape([-1,206,4])
            model = SANPolyA()
            early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
            model.fit(x=x_train, y=y_train, epochs=100, verbose=0, batch_size=256, callbacks=[early_stopping], validation_data=(x_valid,y_valid))
            
            scores = model.evaluate(x=x_target,y=y_target,batch_size=256,verbose=0)
            x1_target = x1_target.reshape([-1,206,4])
            scores1 = model.evaluate(x=x1_target,y=y1_target,batch_size=256,verbose=0)
            acc1+=float(scores1[1])
            acc+=float(scores[1])
            #print(acc)
            #print(scores[1])
        acc = (acc/5) * 100
        acc1 = (acc1/5) * 100

            
        with open('./results/san.txt','a+') as f:
            f.write('source are %s+%s, target are %s and %s \n'% (DOMAIN1,DOMAIN2,DOMAIN3,DOMAIN4))
            f.write('rounds is %d\n'%(rounds+1))
            f.write('%s accuracy: %.1f%% \n' %(DOMAIN3,acc))
            f.write('%s accuracy: %.1f%% \n\n' %(DOMAIN4,acc1))

        #print('\n\n########################\nFinal result:')
        #print('Training accuracy: %.1f%%' % (train_accuracy))
        #print('Validation accuracy: %.1f%%' % (valid_accuracy))
        #print('d2 accuracy: %.1f%%' % (d2_accuracy))
        #print('d3 accuracy: %.1f%%' % (d3_accuracy))
        #print("human accuracy:%.2f%%"%d1_accuracy)
        #print("Training motif accuracy......")


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
    D1_NEG_PATH = 'data/no_rat/negative/'
    D1_POS_PATH = 'data/no_rat/positive/'
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
    D2_NEG_PATH = 'data/no_rat/negative/'
    D2_POS_PATH = 'data/no_rat/positive/'
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
    D3_NEG_PATH = 'data/no_rat/negative/'
    D3_POS_PATH = 'data/no_rat/positive/'
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
    D4_NEG_PATH = 'data/no_rat/negative/'
    D4_POS_PATH = 'data/no_rat/positive/'
if args.d4== 4:
    DOMAIN4='bovine'
    D4_NEG_PATH = 'data/bovine/negative/'
    D4_POS_PATH = 'data/bovine/positive/'
main()









