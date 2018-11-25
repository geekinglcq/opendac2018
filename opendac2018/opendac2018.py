import multiprocessing as mkl
from settings import local_output_path,pubs_validate_path,CPU_COUNT, pubs_train_path, pos_pair_path
from tools import label2assign

from XMeans import XMeans
from collections import defaultdict
from rules.pos import generate_positive_pair
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from active_semi_clustering.semi_supervised.pairwise_constraints import *
import os
import pickle as pkl
import json
import numpy as np

pubs_validate = json.load(open(pubs_validate_path,'r'))
pubs_train = json.load(open(pubs_train_path, 'r'))
local_output = pkl.load(open(local_output_path, 'rb'))
pos_pair = generate_positive_pair()
cluster_num = json.load(open("./output/cluster_num_17.json", 'r'))
j68 = json.load(open('0.68.json','r'))

def clustering_with_const(name, method='PCKMeans', num_clusters=None):
    """
    args:
      str  method: COPKMeans, PCKMeans, MPCKMeans,
                MPCKMeansMF, MKMeans, RCAKMeans
      list ml: must-link constraints
      list cl: cannot-link constraints
    """

    ids = [p['id'] for p in pubs_validate[name]]
    id2ind = {k : v for v, k in enumerate(ids)}
    must_link = [(id2ind[a], id2ind[b]) for a, b in pos_pair[name]]
    Z = np.array([local_output[id] for id in ids])
    scalar = StandardScaler()
    emb_norm = scalar.fit_transform(Z)

    assert method in set(['COPKMeans', 'PCKMeans', 'MPCKMeans', 'MPCKMeansMF'
                          'MKMeans', 'RCAKMeans'])
    model = eval(method)(n_clusters=len(j68[name]))
    try:
        model.fit(emb_norm, ml=must_link)
    except:
        print("too many cluster: ", name)
        return j68[name]

    return label2assign(ids, model.labels_)


def clustering(name, method='XMeans', num_clusters=None):
    ids = [p['id'] for p in pubs_validate[name]]       #todo: 视情况修改
    Z = np.array([local_output[id] for id in ids])
    scalar = StandardScaler()
    emb_norm = scalar.fit_transform(Z)
    if method == 'XMeans':
        model = XMeans()
        model.fit(emb_norm)

    elif method == 'HAC':
        assert num_clusters is not None
        model = AgglomerativeClustering(n_clusters=num_clusters).fit(emb_norm)
    return label2assign(ids, model.labels_)



if __name__=="__main__":
    p = mkl.Pool(CPU_COUNT)

    ###for train:
    #res = p.starmap(clustering,  zip( pubs_train.keys(), ['XMeans']*len(pubs_train.keys()) ) )
    #J = dict(zip(pubs_train.keys(), res))
    #json.dump(J, open('assignment_train_result.json', 'w'))

    ##for val:
    res = p.starmap(clustering_with_const,  zip( pubs_validate.keys(), ['PCKMeans']*len(pubs_validate.keys()) ) )
    J = dict(zip(pubs_validate.keys(), res))
    json.dump(J, open('assignment_validate_result.json', 'w'))
