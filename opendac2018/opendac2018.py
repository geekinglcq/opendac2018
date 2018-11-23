import multiprocessing as mkl
from settings import local_output_path,pubs_validate_path,CPU_COUNT
from tools import label2assign
from XMeans import XMeans
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import pickle as pkl
import json
import numpy as np

pubs_validate = json.load(open(pubs_validate_path,'r'))


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
    