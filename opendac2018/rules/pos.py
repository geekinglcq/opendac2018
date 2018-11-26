import os
import sys
import json
import numpy as np

from itertools import combinations
from collections import defaultdict
sys.path.append('../')
from settings import pubs_train_path, pubs_validate_path, pos_pair_path, \
    rule_check_file_path, assignments_train_path, CPU_COUNT
import multiprocessing as mlp

# VAL_PATH, VAL_NAME2PUB, OUTPUT_DIR, TRAIN_NAME2PUB,\
#    IDF_THRESHOLD, global_output_path, material_path, idf_path,


def exactly_same_co_author(paper_a, paper_b):
    """
    If two paper share the exactly same co-authors, we assume that they have
    the same author.
    Return: True-Same authors False-Not sure
    """
    co_author_a = set([author['name'] for author in paper_a['authors']])
    co_author_b = set([author['name'] for author in paper_b['authors']])
    if (co_author_a == co_author_b) and (len(co_author_a) >= 2):
        return True
    else:
        return False


def nearly_same_co_author(paper_a, paper_b):
    """
    If two paper share the nearly same co-authors, we assume that they have
    the same author.
    Return: True-Same authors False-Not sure
    """
    co_author_a = set([author['name'] for author in paper_a['authors']])
    co_author_b = set([author['name'] for author in paper_b['authors']])
    len_and = len(co_author_a & co_author_b)
    len_or = len(co_author_a | co_author_b)
    if (len_and / len_or) >= 0.8 and (len_or >= 2):
        return True
    else:
        return False


def at_least_one_same_co_author(paper_a, paper_b):
    """
    If two paper share at least one same co-authors, we assume that they have
    the same author.
    Return: True-Same authors False-Not sure
    """
    co_author_a = set([author['name'] for author in paper_a['authors']])
    co_author_b = set([author['name'] for author in paper_b['authors']])
    len_and = len(co_author_a & co_author_b)
    if len_and >= 2:
        return True
    else:
        return False


def at_least_one_same_org(paper_a, paper_b):
    """
    If two paper share at least one same co-authors, we assume that they have
    the same author.
    Return: True-Same authors False-Not sure
    """
    org_a = set([author['org'] for author in paper_a['authors']])
    org_b = set([author['org'] for author in paper_b['authors']])
    len_and = len(org_a & org_b)
    if len_and >= 1:
        return True
    else:
        return False



def gen_rule_check_file():
    """
    Genterate the file for rule check.
    The file is a dict, key is paper id, value is the cluster of this paper,
    value format name_xxx (e.g. li_ma_1, j_xu_3)
    """
    print("Start to generate rule check file")
    data = json.load(open(assignments_train_path))
    rule_check = {}
    for name, clusters in data.items():
        for cid, cluster in enumerate(clusters):
            for paper in cluster:
                rule_check[paper] = "%s_%d" % (name, cid)

    json.dump(rule_check, open(rule_check_file_path, 'w'))
    print("Done. Results was stored in %s." % (rule_check_file_path))


def check_rule_precision(rule):
    """
    Check the precision of the rule.
    """
    correct_count = .0
    total_count = .0
    wrong = []
    correct = []
    data = json.load(open(pubs_train_path))
    if not os.path.isfile(rule_check_file_path):
        gen_rule_check_file()
    check = json.load(open(rule_check_file_path))

    print("Start to check the rule: %s" % (rule.__name__))
    for idx, (name, papers) in enumerate(data.items()):
        if idx | 20:
            print(idx + 1, '/', len(data))
        for paper_a, paper_b in combinations(papers, 2):
            if_same = check[paper_a['id']] == check[paper_b['id']]
            if rule(paper_a, paper_b):
                total_count += 1
                if if_same:
                    correct_count += 1
                    correct.append((paper_a, paper_b))
                else:
                    wrong.append((paper_a, paper_b))
    print('Rule %s  precision: %.3f' % (rule.__name__, correct_count / total_count))
    return correct, wrong

def work_for(papers):
    pairs = set()
    rules = [exactly_same_co_author]
    for paper_a, paper_b in combinations(papers, 2):
        for rule in rules:
            if rule(paper_a, paper_b):
                pairs.add((paper_a['id'], paper_b['id']))
    return list(pairs)

def generate_positive_pair(mode=0):
    """
    Generate the pair of doc that obey the strong rule, which makes them must
    belong to the same cluster.
    Return: dict: key-name, values-list of pos_pair tuple
    {'name': [(pid,pid), ... , (pid, pid)]}
    """
    if os.path.exists(pos_pair_path):
        return json.load(open(pos_pair_path, 'r'))


    pubs_train = json.load(open(pubs_train_path))
    pubs_validate = json.load(open(pubs_validate_path))
    data = {**pubs_train, **pubs_validate}

    pool = mlp.Pool(CPU_COUNT)
    pos_pair = dict(zip(data.keys(), pool.map(work_for, data.values())))

    json.dump(pos_pair, open(pos_pair_path, 'w'))
    return pos_pair
