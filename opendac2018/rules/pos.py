import os
import sys
import json
import numpy as np

from itertools import combinations
from collections import defaultdict
sys.path.append('../')
from settings import pubs_train_path, pubs_validate_path, pos_pair_train_path, \
    pos_pair_val_path

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
    if co_author_a == co_author_b:
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


def generate_positive_pair(mode=0):
    """
    Generate the pair of doc that obey the strong rule, which makes them must
    belong to the same cluster.
    Return: dict: key-name, values-list of pos_pair tuple
    {'name': [(pid,pid), ... , [pid, pid]]}
    """
    if mode:
        data_path = pubs_train_path
        pos_pair_path = pos_pair_train_path
    else:
        data_path = pubs_validate_path
        pos_pair_path = pos_pair_val_path

    rules = [exactly_same_co_author]
    pos_pair = defaultdict(list)

    data = json.load(open(data_path))
    for idx, (name, papers) in enumerate(data.items()):
        print(name, "%d/%d" % (idx + 1, len(data)))
        pairs = set()
        for paper_a, paper_b in combinations(papers, 2):
            for rule in rules:
                if rule(paper_a, paper_b):
                    pairs.add((paper_a['id'], paper_b['id']))
        pos_pair[name] = list(pairs)

    json.dump(pos_pair, open(pos_pair_path, 'w'))
    return pos_pair
