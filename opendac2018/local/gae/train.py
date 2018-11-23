from __future__ import division
from __future__ import print_function

import os
import sys
import time
import json
import pickle
from os.path import join

# Train on CPU (hide GPU) due to memory constraints

import codecs
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering

sys.path.append('../..')
from local.gae.optimizer import OptimizerAE, OptimizerVAE
from local.gae.input_data import load_local_data
from local.gae.model import GCNModelAE, GCNModelVAE
from local.gae.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight

sys.path.append('../')
from tools import pairwise_precision_recall_f1, cal_f1
from settings import IDF_THRESHOLD, DATA_DIR, OUTPUT_DIR, idf_path, \
                     global_output_path, material_path, local_output_path, TRAIN_NAME2PUB, \
                     VAL_NAME2PUB, VAL_PATH, cuda_visible_devices

# IDF_THRESHOLD = 32
# DATA_DIR = '../../data/'
# OUTPUT_DIR = '../../output/'
# idf_path = 'idf.pkl'  # word    -> idf value, float
# global_output_path = 'global_output.pkl'  # doc_id  -> Y_i, np.ndarray
# material_path = 'material.pkl'  # doc_id  -> [word1, word2, ...], list
# local_output_path = join(OUTPUT_DIR, 'local_output.pkl')  # doc_id -> z_i
# TRAIN_NAME2PUB = join(DATA_DIR, 'assignment_train.json')
# VAL_NAME2PUB = join(DATA_DIR, 'await_validation.json')
# VAL_PATH = join(DATA_DIR, 'pubs_validate.json')

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128,
                     'Number of units in hidden layer 1.')  # 32
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')  # 16
flags.DEFINE_float('weight_decay', 0.,
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('name', 'hui_fang', 'Dataset string.')
# flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('is_sparse', 0, 'Whether input features are sparse.')

model_str = FLAGS.model
name_str = FLAGS.name
start_time = time.time()

local_output = {}

def gae_for_na(name, mode=0):
    """
    train and evaluate disambiguation results for a specific name
    :param name:  author name
    :return: evaluation results
    :mode: 0-train 1-val
    """
    pids, adj, features, labels = load_local_data(name=name)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix(
        (adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train = gen_train_edges(adj)

    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    input_feature_dim = features.shape[1]
    if FLAGS.is_sparse:  # TODO to test
        # features = sparse_to_tuple(features.tocoo())
        # features_nonzero = features[1].shape[0]
        features = features.todense()  # TODO
    else:
        features = normalize_vectors(features)

    # Define placeholders
    placeholders = {
        # 'features': tf.sparse_placeholder(tf.float32),
        'features': tf.placeholder(
            tf.float32, shape=(None, input_feature_dim)),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, input_feature_dim)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, input_feature_dim, num_nodes)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum(
    )  # negative edges/pos edges
    print('positive edge weight', pos_weight)
    norm = adj.shape[0] * adj.shape[0] / float(
        (adj.shape[0] * adj.shape[0] - adj.nnz) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(
                preds=model.reconstructions,
                labels=tf.reshape(
                    tf.sparse_tensor_to_dense(
                        placeholders['adj_orig'], validate_indices=False),
                    [-1]),
                pos_weight=pos_weight,
                norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(
                preds=model.reconstructions,
                labels=tf.reshape(
                    tf.sparse_tensor_to_dense(
                        placeholders['adj_orig'], validate_indices=False),
                    [-1]),
                model=model,
                num_nodes=num_nodes,
                pos_weight=pos_weight,
                norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    def get_embs():
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)  # z_mean is better
        return emb

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features,
                                        placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run(
            [opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
              "{:.5f}".format(avg_cost), "train_acc=",
              "{:.5f}".format(avg_accuracy), "time=",
              "{:.5f}".format(time.time() - t))

    emb = get_embs()

    for idx, pid in enumerate(pids):
        local_output[pid] = emb[idx]

    # Train mode calcul F1
    if not mode:
        n_clusters = len(set(labels))
        emb_norm = normalize_vectors(emb)
        model = AgglomerativeClustering(n_clusters=n_clusters)
        model.fit(emb_norm)
        prec, rec, f1 = pairwise_precision_recall_f1(model.labels_, labels)
        print('pairwise precision', '{:.5f}'.format(prec), 'recall',
              '{:.5f}'.format(rec), 'f1', '{:.5f}'.format(f1))
        return [prec, rec, f1], num_nodes, n_clusters



def load_names(mode=0):
    """ mode: 0-train 1-val
    """
    if mode:
        filename = 'val_name_list.json'
    else:
        filename = 'train_name_list.json'
    return json.load(open(join(DATA_DIR, filename)))


def main(mode=0):
    names = load_names(mode=mode)
    if mode == 0:
        wf = codecs.open(
            join(OUTPUT_DIR, 'local_clustering_results.csv'),
            'w',
            encoding='utf-8')
        wf.write('name,n_pubs,n_clusters,precision,recall,f1\n')
    metrics = np.zeros(3)

    cnt = 0

    if mode:
        for name in names:
            gae_for_na(name, mode=1)
    else:
        for name in names:
            cur_metric, num_nodes, n_clusters = gae_for_na(name, mode=mode)
            wf.write('{0},{1},{2},{3:.5f},{4:.5f},{5:.5f}\n'.format(
                name, num_nodes, n_clusters, cur_metric[0], cur_metric[1],
                cur_metric[2]))
            wf.flush()
            for i, m in enumerate(cur_metric):
                metrics[i] += m
            cnt += 1
            macro_prec = metrics[0] / cnt
            macro_rec = metrics[1] / cnt
            macro_f1 = cal_f1(macro_prec, macro_rec)
            print('average until now', [macro_prec, macro_rec, macro_f1])
            time_acc = time.time() - start_time
            print(cnt, 'names', time_acc, 'avg time', time_acc / cnt)
        macro_prec = metrics[0] / cnt
        macro_rec = metrics[1] / cnt
        macro_f1 = cal_f1(macro_prec, macro_rec)
        wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
            macro_prec, macro_rec, macro_f1))
        wf.close()


if __name__ == '__main__':
    #for train
    main(0)

    #for val
    main(1)
    pickle.dump(local_output, open(local_output_path, 'wb'))

