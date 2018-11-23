import multiprocessing as mkl
from settings import local_output_path,pubs_validate_path,CPU_COUNT, pos_pair_path
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


def clustering_with_const(name, method='PCKMeans', ml=[], cl=[], num_clusters=None):
    """
    args:
      str  method: COPKMeans, PCKMeans, MPCKMeans,
                MPCKMeansMF, MKMeans, RCAKMeans
      list ml: must-link constraints
      list cl: cannot-link constraints
    """

    pubs_validate = json.load(open(pubs_validate_path,'r'))
    local_output = pkl.load(open(local_output_path, 'rb'))
    if not os.path.isfile(pos_pair_path):
        generate_positive_pair()
    pos_pair = json.load(open(pos_pair_path))[name]

    ids = [p['id'] for p in pubs_validate[name]]
    id2ind = {k : v for v, k in enumerate(ids)}
    must_link = [(id2ind[a], id2ind[b]) for a, b in pos_pair]
    Z = np.array([local_output[id] for id in ids])
    scalar = StandardScaler()
    emb_norm = scalar.fit_transform(Z)

    assert method in set(['COPKMeans', 'PCKMeans', 'MPCKMeans', 'MPCKMeansMF'
                          'MKMeans', 'RCAKMeans'])
    model = eval(method)(n_clusters=50)
    model.fit(emb_norm, ml=must_link)

    return label2assign(ids, model.labels_)


def clustering(name, method='XMeans', num_clusters=None):
    ids = [p['id'] for p in pubs_validate[name]]
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
    local_output = pkl.load(open(local_output_path,'rb'))
    p = mkl.Pool(CPU_COUNT)
    res = p.starmap(clustering,  zip( pubs_validate.keys(), ['XMeans']*len(pubs_validate.keys()) ) )
    J = dict(zip(pubs_validate.keys(), res))
    json.dump(J, open('assignment_validate_result.json', 'w'))
