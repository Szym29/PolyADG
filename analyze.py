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
def transfer_back(fre):
    #['A', 'G', 'T', 'C']
    output = []
    for each in fre:
        temp_list = []
        for i in each:
            temp = ''
            for j in i:

                if np.argmax(j) ==0:
                    temp+='A'
                elif  np.argmax(j) == 1:
                    temp+='G'
                elif  np.argmax(j) == 2:
                    temp+='T'
                elif  np.argmax(j) == 3:
                    temp+='C'
                else:
                    temp+='K'
            temp_list.append(temp)
        output.append(temp_list)
    return output
def visulize_filters(data, filters,tag):
    data = np.reshape(data,[data.shape[0],data.shape[1],4])
    filters = np.reshape(filters,[filters.shape[0],filters.shape[1],16])
    list_filters = list(filters)
    mer = np.argmax(filters, axis=1)
    

    #print(mer)
    fre = []
    for i in range(len(data)):
        temp = []
        for j in range(0,16):
            temp.append(data[i][mer[i][j]:mer[i][j]+10])
        temp = np.array(temp)
        fre.append(temp)

    fre = transfer_back(fre)
    position_value = np.mean(filters,axis=2)
    position_value=np.mean(position_value,axis=0)

    #with open('./visualization/%s+%s+position_value.txt'%(DOMAIN1,DOMAIN2),'a+') as f:
    #    for each in position_value:
    #        f.write(str(each)+'\n')
    for each in fre:
        for i in range(0,16):
    
            with open('./visualization/%s+%s/filters%d-%s.txt'%(DOMAIN1,DOMAIN2,i+1,tag),'a+') as f:
                #f.write('source are %s+%s, target is %s \n'% (D1_POS_PATH,D2_POS_PATH,D3_POS_PATH))
                f.write(str(each[i])+'\n')
def position_value(filters,labels):
    filters = np.reshape(filters,[filters.shape[0],filters.shape[1],16])
    labels = np.argmax(labels,axis=1)
    pos = []
    neg = []
    for i in range(labels.shape[0]):
        if labels[i]==0:
            pos.append(i)
        else:
            neg.append(i)
    pos_po = filters[pos,:]
    neg_po = filters[neg,:]
    
    pos_value = np.mean(pos_po,axis=2)
    pos_value=np.mean(pos_value,axis=0)
    neg_value = np.mean(neg_po,axis=2)
    neg_value=np.mean(neg_value,axis=0)
    position_value={}
    position_value['pos']=pos_value
    position_value['neg']=neg_value

    return position_value
    #position_value = np.mean(filters,axis=2)
    #position_value=np.mean(position_value,axis=0)

def proportion(dataset,labels,fold):
    dataset_folds = np.array_split(dataset,10)
    labels_folds = np.array_split(labels,10)
    dataset = np.concatenate([dataset_folds[i] for i in fold['round']], axis=0)
    labels = np.concatenate([labels_folds[i] for i in fold['round']], axis=0)
    return dataset, labels
    
def pad_dataset(dataset, labels):
    ''' Change dataset height to height + 2*DEPTH - 2'''
    new_dataset = np.ones([dataset.shape[0], dataset.shape[1]+2*PATCH_SIZE-2, dataset.shape[2], dataset.shape[3]], dtype = np.float32) * 0.25
    new_dataset[:, PATCH_SIZE-1:-(PATCH_SIZE-1), :, :] = dataset
    labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
    return new_dataset, labels

def pixel_level_shuffle(data):
    #data = np.reshape(data,(data.shape[0],-1))
    shuffled_data = []
    for each in data:
        each = each[9:-9]
        #each = np.reshape(each,(each.shape[0]*each.shape[0]))
        #print(each.shape)
        
        permutation = np.random.permutation(each.shape[0])

        each = each[permutation,:,:]


        each = np.reshape(each,(each.shape[0]*each.shape[1]*each.shape[2]))

        shuffled_data.append(each)
    #print(data.shape)
    shuffled_data = np.array(shuffled_data)
    shuffled_data = shuffled_data.astype(np.float32)
    #print(shuffled_data.shape)
    #shuffled_data = np.reshape(shuffled_data, (-1, SEQ_LEN * NUM_CHANNELS))
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

            #'tf_learning_rate': 0.001,
            'tf_learning_rate': 0.3,
            'tf_motif_init_weight': rand_log(1e-2, 1),
            'tf_fc_init_weight': rand_log(1e-2, 1),
            'tf_keep_prob': np.random.choice([.5, .75, 1.0]),
            'tf_ngroups': np.random.choice([2, 4, 8]),
            'tf_mlp_init_weight': rand_log(1e-2, 10),
            'tf_concat_init_weight': rand_log(1e-2, 1),
            'tf_keep_prob':0.2,
            #'tf_keep_prob':0.8,
            

        }
    return hyper_dict


def train(dataset, d2,d3,hyper_dict):
    graph = tf.Graph()
    with graph.as_default():

        # Load hyper-params
        tf_learning_rate = hyper_dict['tf_learning_rate']
        tf_keep_prob = hyper_dict['tf_keep_prob']
        tf_mlp_init_weight = hyper_dict['tf_mlp_init_weight']
        tf_concat_init_weight = hyper_dict['tf_concat_init_weight']

        # Input data.
        l = tf.placeholder(tf.float32, [])
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, SEQ_LEN, 1, NUM_CHANNELS))
        tr_shuffle = tf.placeholder(tf.float32, shape=(BATCH_SIZE,  NUM_CHANNELS*206))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
        tf_train_domain = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 3))
        tf_d2_dataset = tf.constant(d2['test_dataset'])
        tf_d2_shuffle =  tf.constant(pixel_level_shuffle(d2['test_dataset']))
        tf_d2_label = tf.constant(
           d2['test_labels'])

        
        tf_d1_dataset = tf.constant(dataset['test_dataset'])
        tf_d1_shuffle =  tf.constant(pixel_level_shuffle(dataset['test_dataset']))
        tf_d1_label = tf.constant(dataset['test_labels'])
           
        
        tf_d3_dataset = tf.constant(np.concatenate([d3['train_dataset'],d3['valid_dataset'],
           d3['test_dataset']]))
        tf_d3_shuffle = tf.constant(pixel_level_shuffle(np.concatenate([d3['train_dataset'],d3['valid_dataset'],
           d3['test_dataset']])))
        tf_d3_label = tf.constant(np.concatenate([d3['train_labels'],d3['valid_labels'],
            d3['test_labels']]))

        tf_motif_d3_dataset = {}
        tf_motif_d3_label = {}

        tf_train_valid_dataset = tf.constant(np.concatenate([dataset['train_dataset'], d2['train_dataset']]))
        tf_train_valid_shuffle =  tf.constant(pixel_level_shuffle(np.concatenate([dataset['train_dataset'], d2['train_dataset']])))
       
        tf_train_valid_label = tf.constant(np.concatenate([dataset['train_labels'], d2['train_labels']]))

        tf_valid_dataset = tf.constant(np.concatenate([dataset['valid_dataset'],d2['valid_dataset'], dataset['test_dataset'],d2['test_dataset']]))
        tf_valid_shuffle = tf.constant(pixel_level_shuffle(np.concatenate([dataset['valid_dataset'],d2['valid_dataset'], dataset['test_dataset'],d2['test_dataset']])))
        tf_valid_label = tf.constant(np.concatenate([dataset['valid_labels'],d2['valid_labels'],dataset['test_labels'],d2['test_labels']]))


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
            [824,128],stddev = tf_mlp_init_weight))
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
        conv1_w = tf.Variable(tf.truncated_normal([3, 1, NUM_CHANNELS, DEPTH], stddev=1e-1))
        conv1_b = tf.Variable(tf.zeros([DEPTH]))


        # Model.

        def model(data, shuffle, label, drop=True):
            
        # MLP
        
            Hex = True
            mlp_1 = tf.nn.relu(tf.matmul(shuffle, mlp_1_weights) + mlp_1_biases)

        # CNN
            conv = tf.nn.conv2d(data, conv_weights, [1, 1, 1, 1], padding='VALID')

            hidden = tf.nn.relu(conv)
            filters = hidden
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
            #print((tf.matmul(model_H, model_H, transpose_a=True)))
            #exit()
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=model_loss))
            if Hex:
                model_loss = tf.nn.l2_normalize(model_loss, 0)
                model_H = tf.nn.l2_normalize(model_H, 0)
                model_loss = model_loss - \
                         tf.matmul(tf.matmul(
                             tf.matmul(model_H, tf.matrix_inverse(tf.matmul(model_H, model_H, transpose_a=True))),
                             model_H, transpose_b=True), model_loss)

                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=model_loss))
                
            return loss, model_pred, filters

    # Training computation.
        loss, _, _ = model(tf_train_dataset, tr_shuffle, tf_train_labels, drop=True)
    # Optimizer.
        global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
        stepOp = tf.assign_add(global_step, 1).op
        learning_rate = tf.train.exponential_decay(tf_learning_rate, global_step, 3000, 0.96)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # Predictions for the training, validation, and test data.
        motif_train_prediction = {}

        train_loss, train_valid,_ = model(tf_train_valid_dataset,tf_train_valid_shuffle, tf_train_valid_label, drop=True)
        train_prediction = tf.nn.softmax(train_valid)
        _,d1,d1_filters = model(tf_d1_dataset,tf_d1_shuffle, tf_d1_label,drop = True)
        d1_prediction = tf.nn.softmax(d1)

        
        valid_loss, validation,filters = model(tf_valid_dataset,tf_valid_shuffle, tf_valid_label, drop=True)
        valid_prediction = tf.nn.softmax(validation)
        
        
        _, d2_out,d2_filters = model(tf_d2_dataset, tf_d2_shuffle,tf_d2_label, drop=True)
        d2_prediction = tf.nn.softmax(d2_out)


        _, d3_out,d3_filters = model(tf_d3_dataset,tf_d3_shuffle, tf_d3_label, drop=True)
        d3_prediction = tf.nn.softmax(d3_out)

    # get kernel
    
# Kick off training
    valid_losses = []
    train_resuts = []
    valid_results = []
    d2_results = []
    d3_results = []
    d1_results = []
    filters_pack = []
    save_weights = []

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        train_dataset = np.concatenate([dataset['train_dataset'],d2['train_dataset']])
        train_labels =  np.concatenate([dataset['train_labels'],d2['train_labels']])
        np.random.seed()
        print('Initialized')
        print('Training accuracy at the beginning: %.1f%%' % accuracy(train_prediction.eval(), np.concatenate([dataset['train_labels'], d2['train_labels']])))
        print('Validation accuracy at the beginning: %.1f%%' % accuracy(valid_prediction.eval(), np.concatenate([dataset['valid_labels'],d2['valid_labels'],dataset['test_labels'],d2['test_labels']])))
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
                feed_dict = {tf_train_dataset: batch_data,tr_shuffle :batch_shuffle, tf_train_labels: batch_labels}
                _, l = session.run(
                    [optimizer, loss], feed_dict=feed_dict)
                session.run(stepOp)
            train_resuts.append(accuracy(train_prediction.eval(), np.concatenate([dataset['train_labels'], d2['train_labels']])))

            valid_pred = valid_prediction.eval()
            print('validation loss', valid_loss.eval())
            valid_losses.append(valid_loss.eval())
            valid_results.append(accuracy(valid_pred,  np.concatenate([dataset['valid_labels'],d2['valid_labels'],dataset['test_labels'],d2['test_labels']])))
            d1_pred = d1_prediction.eval()
            d1_results.append(accuracy(d1_pred,dataset['test_labels']))
            d2_pred = d2_prediction.eval()
            d2_results.append(accuracy(d2_pred, d2['test_labels']))

            d3_pred = d3_prediction.eval()
            d3_results.append(accuracy(d3_pred, np.concatenate([d3['train_labels'],d3['valid_labels'],
            d3['test_labels']])))

            #filters_eval = filters.eval()
            #filters_pack.append(filters_eval)
            print('Training accuracy at epoch %d: %.1f%%' % (epoch, train_resuts[-1]))
            print('Validation accuracy: %.1f%%' % valid_results[-1])
            print('target accuracy:%.1f%%'%d3_results[-1])

        # Early stopping based on validation results
            if epoch > 10 and valid_results[-11] > max(valid_results[-10:]):
                train_resuts = train_resuts[:-10]
                valid_results = valid_results[:-10]
                d1_results = d1_results[:-10]
                d2_results = d2_results[:-10]
                d3_results = d3_results[:-10]
                
                #filters = filters
                d1_filter = d1_filters.eval()
                d2_filter = d2_filters.eval()
                d3_filter = d3_filters.eval()
                visulize_filters(dataset['test_dataset'],d1_filter,DOMAIN1)
                visulize_filters(d2['test_dataset'],d2_filter,DOMAIN2)
                visulize_filters(np.concatenate([d3['train_dataset'],d3['valid_dataset'],d3['test_dataset']]),d3_filter,DOMAIN3)
                filters = filters.eval()
                valid_data = np.concatenate([dataset['valid_dataset'],d2['valid_dataset'], dataset['test_dataset'],d2['test_dataset']])
                valid_label = np.concatenate([dataset['valid_labels'],d2['valid_labels'], dataset['test_labels'],d2['test_labels']])
                visulize_filters(valid_data,filters,tag='all')
                position_values =position_value(filters,valid_label) 
                # 
                return train_resuts, valid_results, d2_results, d3_results, valid_losses,d1_results, position_values

            else:

                #filters = filters
                d1_filter = d1_filters.eval()
                d2_filter = d2_filters.eval()
                d3_filter = d3_filters.eval()
                visulize_filters(dataset['test_dataset'],d1_filter,DOMAIN1)
                visulize_filters(d2['test_dataset'],d2_filter,DOMAIN2)
                visulize_filters(np.concatenate([d3['train_dataset'],d3['valid_dataset'],d3['test_dataset']]),d3_filter,DOMAIN3)
                filters = filters.eval()
                valid_data = np.concatenate([dataset['valid_dataset'],d2['valid_dataset'], dataset['test_dataset'],d2['test_dataset']])
                valid_label = np.concatenate([dataset['valid_labels'],d2['valid_labels'], dataset['test_labels'],d2['test_labels']])
                visulize_filters(valid_data,filters,tag='all')
                position_values =position_value(filters,valid_label) 
    return train_resuts, valid_results, d2_results, d3_results,valid_losses,d1_results, position_values


def main():
    try:
        os.system('mkdir ./visualization/%s+%s/'%(DOMAIN1,DOMAIN2))
    except:
        pass
    for rounds in range(0,10):
        print('rounds:%d'%rounds)
        hyper_dict = gen_hyper_dict(HYPER_DICT)
        d1_pos_data, d1_pos_labels, d1_neg_data, d1_neg_labels = hm.produce_dataset(NUM_FOLDS, D1_POS_PATH,D1_NEG_PATH)
        d2_pos_data, d2_pos_labels, d2_neg_data, d2_neg_labels = ms.produce_dataset(NUM_FOLDS,D2_POS_PATH,D2_NEG_PATH)
        d3_pos_data, d3_pos_labels, d3_neg_data, d3_neg_labels = ms.produce_dataset(NUM_FOLDS,D3_POS_PATH,D3_NEG_PATH)
        ## change pos neg propotion
        
    # Cross validate
        train_accuracy_split = []
        valid_accuracy_split = []
        d2_accuracy_split = []
        d3_accuracy_split = []
        d1_accuracy_split = []

        folds = {'round':[k for k in range(rounds+1)]}
        for i in range(NUM_FOLDS):
            split =  {
            'train': [(i + j) % NUM_FOLDS for j in range(NUM_FOLDS-2)], 
            'valid': [(i + NUM_FOLDS-2) % NUM_FOLDS], 
            'test': [(i + NUM_FOLDS-1) % NUM_FOLDS]
            }


            split1= {
            'train': [(i + j) % NUM_FOLDS for j in range(NUM_FOLDS-2)], 
            'valid': [(i + NUM_FOLDS-2) % NUM_FOLDS], 
            'test': [(i + NUM_FOLDS-1) % NUM_FOLDS]
            }
            fold60 = {'round':[k for k in range(0,6)]}
            fold100 = {'round':[k for k in range(0,10)]}
            
            
            d1_temp = hm.data_split(d1_pos_data, d1_pos_labels, d1_neg_data, d1_neg_labels, NUM_FOLDS, split,folds)
            
            d2_temp = ms.data_split(d2_pos_data, d2_pos_labels, d2_neg_data, d2_neg_labels, NUM_FOLDS,
                                          split1,folds)
            d3_temp = ms.data_split(d3_pos_data, d3_pos_labels, d3_neg_data, d3_neg_labels, NUM_FOLDS,
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


            position_pos_value_split=[]
            position_neg_value_split=[]
            train_resuts, valid_results, d2_results, d3_results, valid_loss,d1_results,position_values = train(d1, d2, d3,hyper_dict)

            position_pos_value_split.append(position_values['pos'])
            position_neg_value_split.append(position_values['neg'])
            train_accuracy_split.append(train_resuts[-1])
            valid_accuracy_split.append(valid_results[-1])
            d2_accuracy_split.append(d2_results[-1])
            d3_accuracy_split.append(d3_results[-1])
            d1_accuracy_split.append(d1_results[-1])


        train_accuracy = np.mean(train_accuracy_split)
        valid_accuracy = np.mean(valid_accuracy_split)
        d2_accuracy = np.mean(d2_accuracy_split)
        d3_accuracy = np.mean(d3_accuracy_split)
        d1_accuracy = np.mean(d1_accuracy_split)

        position_pos_values = np.mean(np.array(position_pos_value_split),axis=0)
        position_neg_values = np.mean(np.array(position_neg_value_split),axis=0)
 

        with open('./visualization/%s+%s/position_pos_value.txt'%(DOMAIN1,DOMAIN2),'a+') as f:
            for each in position_pos_values:
                f.write(str(each)+'\n')
        with open('./visualization/%s+%s/position_neg_value.txt'%(DOMAIN1,DOMAIN2),'a+') as f:
            for each in position_neg_values:
                f.write(str(each)+'\n')
        #with open('./results/polyadg.txt','a+') as f:
        #    f.write('source are %s+%s, target is %s \n'% (D1_POS_PATH,D2_POS_PATH,D3_POS_PATH))
        #    f.write('rounds is %d\n'%(rounds+1))
        #    f.write('accuracy: %.1f%% \n\n' %d3_accuracy)





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










