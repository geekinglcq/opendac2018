import os
import sys
import json
import pickle
from numpy.random import shuffle
from os.path import join
import multiprocessing as mlp

sys.path.append('../')
from settings import VAL_PATH, VAL_NAME2PUB, OUTPUT_DIR, TRAIN_NAME2PUB,\
    IDF_THRESHOLD, global_output_path, material_path, idf_path, CPU_COUNT
print(VAL_PATH)


def gen_validation_name_to_pubs():
    val_data = json.load(open(VAL_PATH))
    val_name2pub = {}
    for name, pubs in val_data.items():
        val_name2pub[name] = []
        for pub in pubs:
            val_name2pub[name].append(pub["id"])
    json.dump(val_name2pub, open(VAL_NAME2PUB, 'w'))


def get_common_score_DP(set1, set2, idf):
    s = [sorted(set1), sorted(set2)]
    I = (0 if len(s[0])<len(s[1]) else 1)
    J = -1
    idf_sum = 0
    for i in range(len(s[I])):
        for j in range(J+1, len(s[1-I])):
            if s[I][i].startswith(s[1-I][j]) or s[1-I][j].startswith(s[I][i]):
                idf_sum+=(idf.get(s[I][i], IDF_THRESHOLD) + idf.get(s[1-I][j], IDF_THRESHOLD))/2
                J = j
    return idf_sum

def get_common_score_inters(set1, set2, idf):
    common_features = set1.intersection(set2)
    idf_sum = 0
    for f in common_features:
        idf_sum += idf.get(f, IDF_THRESHOLD)
    return idf_sum

def get_common_score_similar(set1, set2, idf):
    def cos_sim(v1, v2):
        return 0

def old_gen_data_for(name):
   if not os.path.isfile(pos_pair_path):
        pos_pair_path = None
    else:
        pos_pairs = json.load(open(pos_pair_path))
    cur_name_pids = name_to_pubs_test[name]
    pids_set = set()
    pids = []
    pids2label = {}

    # generate content
    with open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w') as wf_content:
        if mode:
            for i, pid in enumerate(cur_name_pids):
                pids2label[pid] = i
                pids.append(pid)
        else:
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

    # generate network
    pids_filter = list(pids_set)
    n_pubs = len(pids_filter)
    edges = set()
    with open(join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w') as wf_network:
        for i in range(n_pubs - 1):
            author_feature1 = set(lc_feature.get(pids_filter[i]))
            for j in range(i + 1, n_pubs):
                author_feature2 = set(lc_feature.get(pids_filter[j]))
                common_features = author_feature1.intersection(author_feature2)
                idf_sum = 0
                for f in common_features:
                    idf_sum += idf.get(f, idf_threshold)
                    # print(f, idf.get(f, idf_threshold))
                if idf_sum >= idf_threshold:
                    edges.add((pids_filter[i], pids_filter[j]))
        if pos_pair_path:
            edges.update(pos_pairs[name])
        for (pidi, pidj) in edges:
            wf_network.write('{}\t{}\n'.format(pidi, pidj))

def gen_data_for(name):
    cur_name_pids = name_to_pubs_test[name]
    pids_set = set()
    pids = []
    pids2label = {}

    # generate content
    with open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w') as wf_content:
        if mode:
            for i, pid in enumerate(cur_name_pids):
                pids2label[pid] = i
                pids.append(pid)
        else:
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

    # generate network
    pids_filter = list(pids_set)
    n_pubs = len(pids_filter)
    with open(join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w') as wf_network:
        for i in range(n_pubs - 1):
            author_feature1 = set(lc_feature.get(pids_filter[i]))
            for j in range(i + 1, n_pubs):
                author_feature2 = set(lc_feature.get(pids_filter[j]))
                idf_sum = get_common_score_DP(author_feature1, author_feature2, idf)
                if idf_sum >= IDF_THRESHOLD:
                    wf_network.write('{}\t{}\n'.format(pids_filter[i], pids_filter[j]))


if __name__ == '__main__':
    idf = pickle.load(open(join(OUTPUT_DIR, idf_path), 'rb'))
    lc_inter = pickle.load(open(join(OUTPUT_DIR, global_output_path), 'rb'))
    lc_feature = pickle.load(open(join(OUTPUT_DIR, material_path), 'rb'))
    graph_dir = join(OUTPUT_DIR, 'graph-{}'.format(IDF_THRESHOLD))
    os.makedirs(graph_dir, exist_ok=True)

    # for train
    mode = 0
    name_to_pubs_test = json.load(open(TRAIN_NAME2PUB))
    pos_pair_path = pos_pair_train_path
    p = mlp.Pool(CPU_COUNT)
    p.map(gen_data_for, name_to_pubs_test.keys())
    p.close()

    # for validate
    mode = 1
    if not os.path.exists(VAL_NAME2PUB):
        gen_validation_name_to_pubs()
    name_to_pubs_test = json.load(open(VAL_NAME2PUB))
    pos_pair_path = pos_pair_val_path
    p = mlp.Pool(CPU_COUNT)
    p.map(gen_data_for, name_to_pubs_test.keys())
    p.close()
