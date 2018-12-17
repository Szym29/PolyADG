import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import sys, os
from decimal import Decimal
from ms_prep import *

############# Model Selection ############
POS_PATH = 'data/mouse/sp_mouse/positive/'
NEG_PATH = 'data/mouse/sp_mouse/negative/'
#POS_PATH = 'data/mouse/bl_mouse/positive/'
#NEG_PATH = 'data/mouse/bl_mouse/negative/'
BATCH_SIZE = 64
PATCH_SIZE = 10
DEPTH = 16
NUM_HIDDEN = 64
SEQ_LEN = 206 + 2*PATCH_SIZE-2
NUM_CHANNELS = 4
NUM_LABELS = 2
NUM_EPOCHS = 400
NUM_FOLDS = 5
HYPER_DICT = None
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
            'tf_learning_rate': 0.004793020077587054,
            'tf_momentum': 0.9829484057595264,
            'tf_motif_init_weight': 5.731596189331555,
            'tf_fc_init_weight': 0.02237206407956654,
            'tf_motif_weight_decay': 0.0001518983965598543,
            'tf_fc_weight_decay': 3.122313707573144e-05,
            'tf_keep_prob': 0.5,
            'tf_ngroups': 4,
            'tf_mlp_init_weight': rand_log(1e-2,10),
            'tf_concat_init_weight': rand_log(1e-2,10)
        }


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


def train(dataset, hyper_dict):
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
        tf_mlp_init_weight = hyper_dict['tf_mlp_init_weight']
        tf_concat_init_weight = hyper_dict['tf_concat_init_weight']


        # Input data.
        tf_train_dataset = tf.placeholder(
          tf.float32, shape=(BATCH_SIZE, SEQ_LEN, 1, NUM_CHANNELS))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
        tf_valid_dataset = tf.constant(dataset['valid_dataset'])
        tf_valid_label = tf.constant(dataset['valid_labels'])
        tf_test_dataset = tf.constant(dataset['test_dataset'])
        tf_test_label =  tf.constant(dataset['test_labels'])
        tf_motif_test_dataset = {}
        tf_motif_test_label = {}
        for motif in MOUSE_MOTIF_VARIANTS:
            tf_motif_test_dataset[motif] = tf.constant(dataset['motif_dataset'][motif]['test_dataset'])
            tf_motif_test_label[motif] = tf.constant(dataset['motif_dataset'][motif]['test_labels'])

        # Variables.
        conv_weights = tf.Variable(tf.truncated_normal(
          [PATCH_SIZE, 1, NUM_CHANNELS, DEPTH], stddev=1e-1))
        conv_biases = tf.Variable(tf.zeros([DEPTH]))
        layer1_weights = tf.Variable(tf.truncated_normal(
          [21*DEPTH, NUM_HIDDEN], stddev=1e-1))
        layer1_biases = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [NUM_HIDDEN, NUM_LABELS], stddev=1e-1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

        mlp_1_weights = tf.Variable(tf.truncated_normal(
            [896,3000],stddev = tf_mlp_init_weight))
        mlp_2_weights = tf.Variable(tf.truncated_normal(
            [3000, 512],stddev = tf_mlp_init_weight))
        mlp_3_weights = tf.Variable(tf.truncated_normal(
            [512, 32],stddev = tf_mlp_init_weight))
        concat_weights = tf.Variable(tf.truncated_normal(
            [32 + NUM_HIDDEN, NUM_LABELS],stddev = tf_concat_init_weight))
        mlp_1_biases = tf.Variable(tf.constant(1.0, shape=[3000]))
        mlp_2_biases = tf.Variable(tf.constant(1.0, shape=[512]))
        mlp_3_biases = tf.Variable(tf.constant(1.0, shape=[32]))
        concat_biases = tf.Variable(tf.constant(1.0, shape = [2]))
        #w = pickle.load(open('weights/cv3_model.pkl','rb'))    #load weights
        #print('chekchehf',len(w))
        #conv_weights = tf.Variable(tf.cast(w['conv_weights'],dtype=tf.float32))
        #conv_biases = tf.Variable(tf.cast(w['conv_biases'],dtype=tf.float32))
        #layer1_weights = tf.Variable(tf.cast(w['layer1_weights'],dtype=tf.float32))
        #layer1_biases = tf.Variable(tf.cast(w['layer1_biases'],dtype=tf.float32))
        #layer2_weights =tf.Variable(tf.cast(w['layer2_weights'],dtype=tf.float32))
        #layer2_biases = tf.Variable(tf.cast(w['layer2_biases'],dtype=tf.float32))


        # Store Variables
        weights = {}
        weights['conv_weights'] = conv_weights
        weights['conv_biases'] = conv_biases
        weights['layer1_weights'] = layer1_weights
        weights['layer1_biases'] = layer1_biases
        weights['layer2_weights'] = layer2_weights
        weights['layer2_biases'] = layer2_biases
        weights['mlp_1_weights'] = mlp_1_weights
        weights['mlp_2_weights'] = mlp_2_weights
        weights['mlp_3_weights'] = mlp_3_weights
        weights['mlp_1_biases'] = mlp_1_biases
        weights['mlp_2_biases'] = mlp_2_biases
        weights['mlp_3_biases'] = mlp_3_biases
        weights['concat_weights'] = concat_weights
        weights['concat_biases'] = concat_biases



        # Model.
        def model(data, label, Hex = True, drop=True):
            #MLP
            mlp_1 = tf.nn.relu(tf.matmul(tf.reshape(data,[data.shape[0], -1]),mlp_1_weights) + mlp_1_biases)
            mlp_2 = tf.nn.relu(tf.matmul(mlp_1,mlp_2_weights) + mlp_2_biases)
            mlp_3 = tf.nn.relu(tf.matmul(mlp_2,mlp_3_weights) + mlp_3_biases)

            #CNN
            conv = tf.nn.conv2d(data, conv_weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.reshape(conv, [-1, 215, 1, DEPTH//tf_ngroups, tf_ngroups])
            mu, var = tf.nn.moments(conv, [1, 2, 3], keep_dims=True)
            conv = (conv - mu) / tf.sqrt(var + 1e-12)
            conv = tf.reshape(conv, [-1, 215, 1, DEPTH])
            hidden = tf.nn.relu(conv + conv_biases)
            hidden = tf.nn.max_pool(hidden, [1, 10, 1, 1], [1, 10, 1, 1], padding = 'VALID')
            shape = hidden.get_shape().as_list()
            motif_score = tf.reshape(hidden, [shape[0], shape[1]*DEPTH])
            if drop:
                hidden_nodes = tf.nn.dropout(tf.nn.relu(tf.matmul(motif_score, layer1_weights) + layer1_biases), tf_keep_prob)
            else:
                hidden_nodes = tf.nn.relu(tf.matmul(motif_score, layer1_weights) + layer1_biases)


            concat_loss = tf.concat([hidden_nodes, mlp_3], 1)

            pad = tf.zeros_like(mlp_3, tf.float32)

            concat_pred = tf.concat([hidden_nodes, pad], 1)

            pad2 = tf.zeros_like(hidden_nodes, tf.float32)
            concat_H = tf.concat([pad2, mlp_3], 1)
            model_loss = tf.matmul(concat_loss, concat_weights) + concat_biases
            model_pred = tf.matmul(concat_pred, concat_weights) + concat_biases
            model_H = tf.matmul(concat_H, concat_weights) + concat_biases


            if Hex :
                model_loss = tf.nn.l2_normalize(model_loss, 0)
                model_H = tf.nn.l2_normalize(model_H, 0)
                model_loss = model_loss -\
                             tf.matmul(tf.matmul(tf.matmul(model_H, tf.matrix_inverse(tf.matmul(model_H,model_H, transpose_a= True))), model_H, transpose_b = True), model_loss)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = model_loss))
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = concat_loss))
            return loss, model_pred

        # Training computation.
        loss,_ = model(tf_train_dataset,tf_train_labels)
        # Optimizer.
        global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
        stepOp = tf.assign_add(global_step, 1).op
        learning_rate = tf.train.exponential_decay(tf_learning_rate, global_step, 3000, 0.96)
        optimizer = tf.train.MomentumOptimizer(learning_rate, tf_momentum).minimize(loss)

        # Predictions for the training, validation, and test data.


        _, validation = model(tf_valid_dataset, tf_valid_label, drop=False)
        valid_prediction = tf.nn.softmax(validation)

        _, test = model(tf_test_dataset, tf_test_label, drop=False)
        test_prediction = tf.nn.softmax(test)
        motif_test_prediction = {}
        for motif in MOUSE_MOTIF_VARIANTS:
            _, motif_test = model(tf_motif_test_dataset[motif],tf_motif_test_label[motif], drop=False)
            motif_test_prediction[motif] = tf.nn.softmax(motif_test)


    # Kick off training
    valid_results = []
    test_results = []
    motif_test_results = {motif: [] for motif in MOUSE_MOTIF_VARIANTS}
    save_weights = []
    with tf.Session(graph=graph) as session,tf.device('/gpu:0'):
        tf.global_variables_initializer().run()
        train_dataset = dataset['train_dataset']
        train_labels = dataset['train_labels']
        np.random.seed()
        print('Initialized')
        print('Validation accuracy at the beginning: %.1f%%' % accuracy(valid_prediction.eval(), dataset['valid_labels']))
        for epoch in range(NUM_EPOCHS):
            permutation = np.random.permutation(train_labels.shape[0])
            shuffled_dataset = train_dataset[permutation, :, :]
            shuffled_labels = train_labels[permutation, :]
            for step in range(shuffled_labels.shape[0] // BATCH_SIZE):
                offset = step * BATCH_SIZE
                batch_data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
                batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                _, l = session.run(
                [optimizer, loss], feed_dict=feed_dict)
                session.run(stepOp)

            valid_pred = valid_prediction.eval()
            valid_results.append(accuracy(valid_pred, dataset['valid_labels']))
            test_pred = test_prediction.eval()
            test_results.append(accuracy(test_pred, dataset['test_labels']))
            for motif in MOUSE_MOTIF_VARIANTS:
                motif_test_pred = motif_test_prediction[motif].eval()
                motif_test_results[motif].append(accuracy(motif_test_pred, dataset['motif_dataset'][motif]['test_labels']))
            print('Valid accuracy at epoch %d: %.1f%%' % (epoch, valid_results[-1]))

            # Early stopping based on validation results
            if epoch > 10 and valid_results[-11] >= max(valid_results[-10:]):
                valid_results = valid_results[:-10]
                test_results = test_results[:-10]
                motif_test_results = {motif: motif_test_results[motif][:-10] for motif in MOUSE_MOTIF_VARIANTS}
                return valid_results, test_results, motif_test_results, save_weights[0]

            # Model saving
            sw = {}
            for k in weights:
                sw[k] = weights[k].eval()
            if epoch < 10:
                save_weights.append(sw)
            else:
                save_weights.append(sw)
                save_weights.pop(0)

    return valid_results, test_results, motif_test_results, save_weights[-1]


def main(_):

    # block_print()

    hyper_dict = gen_hyper_dict(HYPER_DICT)
    pos_data, pos_labels, neg_data, neg_labels = produce_motif_dataset(NUM_FOLDS, POS_PATH, NEG_PATH)

    # Cross validate
    valid_accuracy_split = []
    test_accuracy_split = []
    motif_test_accuracy_split = {motif: [] for motif in MOUSE_MOTIF_VARIANTS}

    for i in range(NUM_FOLDS):
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
        for motif in MOUSE_MOTIF_VARIANTS:
            dataset['motif_dataset'][motif] = {}
            dataset['motif_dataset'][motif]['test_dataset'], dataset['motif_dataset'][motif]['test_labels'] = pad_dataset(save['motif_dataset'][motif]['test_dataset'], save['motif_dataset'][motif]['test_labels'])

        valid_results, test_results, motif_test_results, save_weights = train(dataset, hyper_dict)
        print("\nbest valid epoch: %d"%(len(valid_results)-1))
        print("Test accuracy: %.2f%%"%test_results[-1])
        print("Validation accuracy: %.2f%%"%valid_results[-1])
        for motif in MOUSE_MOTIF_VARIANTS:
            print('*%s* accuracy: %.1f%%' % (motif, motif_test_results[motif][-1]))

        # Dump model
        if FLAGS.train_dir is not None:
            with open(os.path.join(FLAGS.training_result_dir, 'cv%d_model.pkl'%i), 'wb') as f:
                pickle.dump(save_weights, f, 2)

        valid_accuracy_split.append(valid_results[-1])
        test_accuracy_split.append(test_results[-1])
        for motif in MOUSE_MOTIF_VARIANTS:
            motif_test_accuracy_split[motif].append(motif_test_results[motif][-1])

    valid_accuracy = np.mean(valid_accuracy_split)
    test_accuracy = np.mean(test_accuracy_split)
    motif_test_accuracy = {}
    for motif in MOUSE_MOTIF_VARIANTS:
        motif_test_accuracy[motif] = np.mean(motif_test_accuracy_split[motif])
    print('\n\n########################\nFinal result:')
    print('Validation accuracy: %.1f%%' % (valid_accuracy))
    print('Test accuracy: %.1f%%' % (test_accuracy ))
    for motif in MOUSE_MOTIF_VARIANTS:
        print('*%s* accuracy: %.1f%%' % (motif, motif_test_accuracy[motif]))

    if FLAGS.training_result_dir is not None:
        with open(os.path.join(FLAGS.training_result_dir, 'result.pkl'), 'wb') as f:
            hyper_dict['valid_accuracy'] = valid_accuracy
            hyper_dict['test_accuracy'] = test_accuracy
            hyper_dict['motif_test_accuracy'] = motif_test_accuracy
            pickle.dump(hyper_dict, f, 2)


    # enable_print()


if __name__ == '__main__':
    tf.app.run()









