#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: count_test.py
#Author: chi xiao
#Mail: 
#Created Time:
############################

from os.path import join
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
import json
import pickle
import keras
from keras.layers import Dense, Dropout, LSTM, Bidirectional

count_model_parameters_path = "./model/count_model.h5"


assignment_train_path = "./data/assignment_train.json"
pubs_validate_path = "./data/pubs_validate.json"
pubs_validate_cluster_path = "./output/validate_cluster_num.json"
paper_feature_xi_path = "./output/weighted_embedding.pkl"

paper_feature = {}

with open(paper_feature_xi_path,'rb') as f:
    paper_feature = pickle.load(f)

data_cache = {}

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def root_mean_log_squared_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))
def create_model():
    #model = keras.models.load_model(count_model_parameters_path)
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(300, 100)))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss="msle",
                  optimizer='rmsprop',
                  metrics=[root_mean_squared_error, "accuracy", "msle", root_mean_log_squared_error])
    model.load_weights(count_model_parameters_path)
    #return model
    return model

def test_validate(model,k=300,flatten=False):
    print ("predict cluster number ...")
    with open(pubs_validate_path,'r') as f:
        pubs_validate_dict = json.load(f)

    author_paper_dict = {}
    for author,papers in pubs_validate_dict.items():
        author_paper_dict.setdefault(author,[])
        for paper in papers:
            author_paper_dict[author].append(paper['id'])

    xs = []
    names = []
    for name in author_paper_dict.keys():
        names.append(name)
        x = []
        items = author_paper_dict[name]
        #print(items)
        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)]
        for p in sampled_points:
            x.append(paper_feature[p])
        if flatten:
            xs.append(np.sum(x, axis=0))
        else:
            xs.append(np.stack(x))
    xs = np.stack(xs)
    kk = model.predict(xs)
    return names, kk

if __name__=="__main__":
    model = create_model()
    names,kk = test_validate(model)
    kk = map(int,np.squeeze(kk))
    result = dict(zip(names,kk))
    print (result)
    with open(pubs_validate_cluster_path,'w') as f:
        json.dump(result,f)
