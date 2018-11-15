# coding: utf-8

# Imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from nltk.corpus import stopwords
import json
import numpy as np
from sklearn.cluster import KMeans
from scipy import sparse
import multiprocessing as mlp
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import calinski_harabaz_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.stem.porter import PorterStemmer  #todo: 还有其他词干抽取器
from collections import defaultdict
import math
import pickle as pkl
import gc
import os
from itertools import chain
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
from keras.callbacks import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

assignments_train_path = './data/assignment_train.json'
pubs_train_path = './data/pubs_train.json'
pubs_validate_path = './data/pubs_validate.json'

## 中间输出文件
material_path = './output/material.pkl'  # doc_id  -> [word1, word2, ...], list
word2vect_model_path = './output/word.emb'  # word2vec model.  usage: Word2Vec.load(...)
idf_path = './output/idf.pkl'  # word    -> idf value, float
weighted_embedding_path = './output/weighted_embedding.pkl'  # doc_id  -> X_i, np.ndarray
triple_set = './output/triple.pkl'  # 'emb'   -> anchors; 'emb_pos': positive weighted embedding; 'emb_neg': negative ones
global_embedding_path = './output/global_output.pkl'  # doc_id  -> Y_i, np.ndarray

## 直接调用: weighted_embedding(), 会返回material, word2vect_model, idf, X_i四元组。
#Y_i的读global_embedding_path

EMBEDDING_DIM = 100


## Read Data
def read_data():
    assignments_train = json.load(open(assignments_train_path, 'r'))
    pubs_train = json.load(open(pubs_train_path, 'r'))
    pubs_validate = json.load(open(pubs_validate_path, 'r'))
    pubs = {**pubs_train, **pubs_validate}
    assert (len(pubs) == len(pubs_train) + len(pubs_validate))
    print('read done!')
    return assignments_train, pubs_train, pubs_validate, pubs


## Word2Vec
def clean_name(nm):
    return re.sub('[^a-z]', '', nm.lower())


def is_same_name(s1, s2):
    return clean_name(s1) == clean_name(s2)


def clean_sent(s, prefix):
    '''
    为区别各字段，不同字段前的词加不同的前缀
    '''
    words = re.sub('[^ \-_a-z]', ' ', s.lower()).split()
    stemer = PorterStemmer()
    return ['__%s__%s' % (prefix, stemer.stem(w)) for w in words]


def ExtractTxt(doc, primary_author):
    """
    把一个文档变为：
    [题目，合作者(姓名,组织)，期刊，摘要，关键词]
    各种预处理之后的word list
    """
    title = clean_sent(doc['title'], 'T') if doc.get('title', None) else []
    venue = clean_sent(doc['venue'], 'V') if doc.get('venue', None) else []
    abstract = clean_sent(doc['abstract'], 'A') if doc.get('abstract',
                                                           None) else []
    keywords = clean_sent(' '.join(doc['keywords']), 'K') if doc.get(
        'keywords', None) else []
    coauthors = []
    if doc.get('authors', None):
        for aut in doc['authors']:
            if not is_same_name(aut.get('name', ''), primary_author):
                coauthors.append(clean_name(aut.get('name', '')))
                coauthors.extend(clean_sent(aut.get('org', ''), 'O'))
    return title + coauthors + venue + abstract + keywords


def word_embedding():
    if os.path.exists(word2vect_model_path) and os.path.exists(material_path):
        model = Word2Vec.load(word2vect_model_path)
        docs = pkl.load(open(material_path, 'rb'))
        return model, docs

    material = []
    paper_id = []
    pool = mlp.Pool(20)
    for k, v in pubs.items():
        material.extend(pool.starmap(ExtractTxt, zip(v, [k] * len(v))))
        paper_id.extend([doc['id'] for doc in v])
    model = Word2Vec(
        material, size=EMBEDDING_DIM, window=5, min_count=5, workers=20)
    docs = dict(zip(paper_id, material))
    pkl.dump(docs, open(material_path, 'wb'))
    model.save(word2vect_model_path)
    pool.close()
    return model, docs


## Weighted Embedding
#todo: 并行
def calc_idf(material):
    if os.path.exists(idf_path):
        return pkl.load(open(idf_path, 'rb'))
    cnt = defaultdict(int)
    idf = {}
    for doc in material:
        for word in doc:
            cnt[word] += 1
    for k, v in cnt.items():
        idf[k] = math.log(len(material) / v)
    pkl.dump(idf, open(idf_path, 'wb'))
    return idf


def project_embedding(docs, wv, idf):
    if os.path.exists(weighted_embedding_path):
        return pkl.load(open(weighted_embedding_path, 'rb'))

    wei_embed = {}
    for id, doc in docs.items():
        word_vecs = []
        sum_weight = 0.0
        for word in doc:
            if word in wv and word in idf:
                word_vecs.append(wv[word] * idf[word])
                sum_weight += idf[word]
        wei_embed[id] = np.sum(word_vecs, axis=0) / sum_weight
    pkl.dump(wei_embed, open(weighted_embedding_path, 'wb'))
    return wei_embed


#得到 X_i, 这部分还算快， 没有写缓存和并行
#warning: 加权结果可能有点大。
def weighted_embedding():
    model, docs = word_embedding()
    print('word embedding done!')
    idf = calc_idf(docs.values())
    weighted = project_embedding(docs, model.wv, idf)
    print('weighted embedding done!')
    return docs, idf, model, weighted


## Generate Triplet Training Data
def get_neg_id(all_papers, excludes):
    while True:
        i = np.random.choice(len(all_papers))
        if all_papers[i] not in excludes:
            return all_papers[i]


def gen_triple(weighted, sz=1000000):
    if os.path.exists(triple_set):
        d = pkl.load(open(triple_set, 'rb'))
        return d['emb'], d['emb_pos'], d['emb_neg']

    triples = []
    authors = list(assignments_train.keys())
    all_papers = list(set([p['id'] for k, v in pubs_train.items() for p in v]))
    I = 0
    try:
        while True:
            author_papers = assignments_train[authors[I]]
            I += 1
            if I >= len(authors):
                I = 0
            for clust in author_papers:
                if len(clust) <= 1:
                    continue
                for pid in clust:
                    sam = np.random.choice(
                        clust, min(len(clust), 5), replace=False)  #因为平均簇大小是5
                    for pid_pos in sam:
                        triples.append(
                            [pid, pid_pos,
                             get_neg_id(all_papers, clust)])
                        if len(triples) >= sz:
                            raise StopIteration
    except StopIteration as e:
        print(len(triples))

    emb = np.array([weighted[t[0]] for t in triples])
    emb_pos = np.array([weighted[t[1]] for t in triples])
    emb_neg = np.array([weighted[t[2]] for t in triples])
    pkl.dump({
        'emb': emb,
        'emb_pos': emb_pos,
        'emb_neg': emb_neg
    }, open(triple_set, 'wb'))
    return emb, emb_pos, emb_neg


## Triplet Model
def l2Norm(x):
    return K.l2_normalize(x, axis=-1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(
        K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(
        K.maximum(
            K.constant(0),
            K.square(y_pred[:, 0, 0]) - K.square(y_pred[:, 1, 0]) + margin))


def accuracy(_, y_pred):
    return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])


class GlobalModel(object):
    def __init__(self):
        self.save_path = 'GlobalModel.h5'
        emb_anchor = Input(shape=(EMBEDDING_DIM, ), name='anchor_input')
        emb_pos = Input(shape=(EMBEDDING_DIM, ), name='pos_input')
        emb_neg = Input(shape=(EMBEDDING_DIM, ), name='neg_input')

        # shared layers
        layer1 = Dense(128, activation='relu', name='first_emb_layer')
        layer2 = Dense(64, activation='relu', name='last_emb_layer')
        norm_layer = Lambda(l2Norm, name='norm_layer', output_shape=[64])

        encoded_emb = norm_layer(layer2(layer1(emb_anchor)))
        encoded_emb_pos = norm_layer(layer2(layer1(emb_pos)))
        encoded_emb_neg = norm_layer(layer2(layer1(emb_neg)))

        pos_dist = Lambda(
            euclidean_distance,
            name='pos_dist')([encoded_emb, encoded_emb_pos])
        neg_dist = Lambda(
            euclidean_distance,
            name='neg_dist')([encoded_emb, encoded_emb_neg])

        def cal_output_shape(input_shape):
            shape = list(input_shape[0])
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists',
            output_shape=cal_output_shape)([pos_dist, neg_dist])

        self.model = Model(
            [emb_anchor, emb_pos, emb_neg],
            stacked_dists,
            name='triple_siamese')
        self.model.compile(
            loss=triplet_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])
        self.infer = Model(
            inputs=self.model.get_layer('anchor_input').get_input_at(0),
            outputs=self.model.get_layer('norm_layer').get_output_at(0))

    def train(self, X, retrain=True):
        if retrain:
            n_triplets = len(X[0])
            self.model.fit(
                X,
                np.ones((n_triplets, 2)),
                batch_size=200,
                epochs=5,
                shuffle=True,
                validation_split=0.2)
        else:
            self.load()

    def predict(self, X):
        return self.infer.predict(X)

    def save(self):
        self.model.save_weights(self.save_path)

    def load(self):
        self.model.load_weights(self.save_path)


if __name__ == "__main__":
    os.makedirs('./output', exist_ok=True)
    m = GlobalModel()
    assignments_train, pubs_train, pubs_validate, pubs = read_data()
    _, _, _, weighted = weighted_embedding()
    emb, emb_pos, emb_neg = gen_triple(weighted)
    print('gen triple done!')

    m.train([emb, emb_pos, emb_neg], retrain=False)
    #m.train([emb, emb_pos, emb_neg], retrain = True); m.save();

    all_id = [p['id'] for k, papers in pubs.items() for p in papers]
    X = np.array([weighted[id] for id in all_id]).reshape((-1, 100))
    Y = m.predict(X)
    d = dict(zip(all_id, Y))
    pkl.dump(d, open(global_embedding_path, 'wb'))
