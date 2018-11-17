from os.path import join
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
import json
import pickle
import os
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from settings import cuda_visible_devices, assignments_train_path, pubs_validate_path, weighted_embedding_path
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

count_model_parameters_path = "./output/count_model.h5"

paper_feature = {}
with open(weighted_embedding_path,'rb') as f:
    paper_feature = pickle.load(f)

data_cache = {}

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def root_mean_log_squared_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def create_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(300, 100)))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss="msle",
                  optimizer='rmsprop',
                  metrics=[root_mean_squared_error, root_mean_log_squared_error])

    return model


def sampler(clusters, k=300, batch_size=10, min=1, max=300, flatten=False):
    xs, ys = [], []
    for b in range(batch_size):
        num_clusters = np.random.randint(min, max)
        sampled_clusters = np.random.choice(len(clusters), num_clusters, replace=False)
        items = []
        for c in sampled_clusters:
            items.extend(clusters[c])
        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)]
        x = []
        for p in sampled_points:
            if p in data_cache:
                x.append(data_cache[p])
            else:
                print("a")
                x.append(lc.get(p))
        if flatten:
            xs.append(np.sum(x, axis=0))
        else:
            xs.append(np.stack(x))
        ys.append(num_clusters)
    return np.stack(xs), np.stack(ys)


def gen_train(clusters, k=300, batch_size=1000, flatten=False):
    while True:
        yield sampler(clusters, k, batch_size, flatten=flatten)


def gen_test(test_names,assignment_train_dict,k=300, flatten=False):
    xs, ys = [], []
    names = []
    for name in test_names:
        names.append(name)
        num_clusters = len(assignment_train_dict[name])
        x = []
        items = []
        for c in assignment_train_dict[name]:  # one person
            items.extend(c)
        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)]
        for p in sampled_points:
            x.append(paper_feature[p])
        if flatten:
            xs.append(np.sum(x, axis=0))
        else:
            xs.append(np.stack(x))
        ys.append(num_clusters)
    xs = np.stack(xs)
    ys = np.stack(ys)
    return names, xs, ys

def run_rnn(k=300, seed=1106):
    train_names, test_names, assignment_train_dict = read_data()
    test_names, test_x, test_y = gen_test(test_names,assignment_train_dict)
    np.random.seed(seed)
    clusters = []
    for name in train_names:
        clusters.extend(assignment_train_dict[name])
    for i, c in enumerate(clusters):
        if i % 100 == 0:
            print(i, len(c), len(clusters))
        for pid in c:
            data_cache[pid] = paper_feature[pid]
    model = create_model()
    early = EarlyStopping('val_loss', patience=5)
    checkpoint = ModelCheckpoint(count_model_parameters_path, 'val_loss', save_best_only=True, save_weights_only=True)
    # print(model.summary())
    model.fit_generator(gen_train(clusters, k=300, batch_size=1000), steps_per_epoch=100, epochs=100,
                        validation_data=(test_x, test_y), callbacks = [early, checkpoint])



def read_data():

    with open(assignments_train_path,'r') as f:
        assignment_train_dict = json.load(f)
    total_names = list(assignment_train_dict.keys())
    train_names = total_names[:80]
    test_names = total_names[80:]
    return train_names,test_names,assignment_train_dict

if __name__ == '__main__':
    run_rnn()
