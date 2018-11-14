import os
import json
import pickle
import numpy as np
from numpy.random import shuffle
from os.path import join

IDF_THRESHOLD = 32
DATA_DIR = './data/'
OUTPUT_DIR = './output/'
idf_path = 'idf.pkl'  # word    -> idf value, float
global_output_path = 'global_output.pkl'  # doc_id  -> Y_i, np.ndarray
material_path = 'material.pkl'  # doc_id  -> [word1, word2, ...], list


def gen_local_data(idf_threshold=IDF_THRESHOLD):
    """
    generate local data (including paper features and paper network)
    for each associated name
    :param idf_threshold: threshold for determining whether there exists an edge
    between two papers (for this demo we set 29)
    """

    name_to_pubs_test = json.load(join(DATA_DIR, 'assignment_train.json'))
    idf = pickle.load(open(join(OUTPUT_DIR, idf_path), 'rb'))
    # INTER_LMDB_NAME = 'author_triplets.emb'
    lc_inter = pickle.load(open(join(OUTPUT_DIR, global_output_path), 'rb'))
    # lc_inter = LMDBClient(INTER_LMDB_NAME)
    # LMDB_AUTHOR_FEATURE = "pub_authors.feature"
    lc_feature = pickle.load(open(join(OUTPUT_DIR, material_path), 'rb'))
    graph_dir = join(OUTPUT_DIR, 'graph-{}'.format(idf_threshold))
    os.makedirs(graph_dir, exist_ok=True)
    for i, name in enumerate(name_to_pubs_test):
        print(i, name)
        cur_name_pids = name_to_pubs_test[name]
        pids_set = set()
        pids = []
        pids2label = {}

        # generate content
        wf_content = open(
            join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
        for i, cur_person_pids in enumerate(cur_name_pids):
            for pid in cur_person_pids:
                pids2label[pid] = i
                pids.append(pid)

        shuffle(pids)
        for pid in pids:
            cur_pub_emb = lc_inter.get(pid)
            if cur_pub_emb is not None:
                cur_pub_emb = list(map(str, cur_pub_emb))
                pids_set.add(pid)
                wf_content.write('{}\t'.format(pid))
                wf_content.write('\t'.join(cur_pub_emb))
                wf_content.write('\t{}\n'.format(pids2label[pid]))
        wf_content.close()

        # generate network
        pids_filter = list(pids_set)
        n_pubs = len(pids_filter)
        print('n_pubs', n_pubs)
        wf_network = open(
            join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w')
        for i in range(n_pubs - 1):
            if i % 10 == 0:
                print(i)
            author_feature1 = set(lc_feature.get(pids_filter[i]))
            for j in range(i + 1, n_pubs):
                author_feature2 = set(lc_feature.get(pids_filter[j]))
                common_features = author_feature1.intersection(author_feature2)
                idf_sum = 0
                for f in common_features:
                    idf_sum += idf.get(f, idf_threshold)
                    # print(f, idf.get(f, idf_threshold))
                if idf_sum >= idf_threshold:
                    wf_network.write('{}\t{}\n'.format(pids_filter[i],
                                                       pids_filter[j]))
        wf_network.close()


if __name__ == '__main__':
    gen_local_data(idf_threshold=IDF_THRESHOLD)
    print('done')
