from XMeans import XMeans
import multiprocessing as mkl
from settings import local_output_path
from tools import label2assign
from settings import pubs_validate_path

def Cluster4(name):
    ids = [p['id'] for p in pubs_validate[name]]
    Z = np.array([local_output[id] for id in ids])
    m = XMeans()
    m.fit(Z)
    return label2assign(ids, m.labels_)


def label2assign(id, y_pred):
    '''
    传入paper id 及预测簇编号
    返回assignment形式：[[id1, id2, ...], [id1, id2, ...]]
    '''
    d = defaultdict(list)
    for i in range(len(id)):
        d[y_pred[i]].append(id[i])
    return list(d.values())


if __name__=="__main__":
    local_output = pkl.load(open(local_output_path,'rb'))
    p = mkl.Pool(10)
    m = XMeans(k_max = 350, max_iter = 1500)
    res = p.map(Cluster4, pubs_validate.keys())
    J = dict(zip(pubs_validate.keys(), res))
    json.dump(J, open('assignment_validate_result.json', 'w'))
    