from os.path import abspath, dirname, join
import os

PROJ_DIR = join(abspath(dirname(__file__)), '.')
DATA_DIR = join(PROJ_DIR, './data/')
OUTPUT_DIR = join(PROJ_DIR, './output/')

TRAIN_NAME2PUB = join(DATA_DIR, 'assignment_train.json')
VAL_NAME2PUB = join(DATA_DIR, 'await_validation.json')
VAL_PATH = join(DATA_DIR, 'pubs_validate.json')
IDF_THRESHOLD = 32
cuda_visible_devices = '1'
local_output_path       = join(OUTPUT_DIR, 'local_output.pkl')  # doc_id  -> Z_i, np.ndarray


# Global settings
assignments_train_path  = join(DATA_DIR, 'assignment_train.json')
pubs_train_path         = join(DATA_DIR, 'pubs_train.json')
pubs_validate_path      = join(DATA_DIR, 'pubs_validate.json')
stop_words_path         = './data/stop_words.txt'
idf_path                = join(OUTPUT_DIR, 'idf.pkl')  # word    -> idf value, float
global_output_path      = join(OUTPUT_DIR, 'global_output.pkl')  # doc_id  -> Y_i, np.ndarray
material_path           = join(OUTPUT_DIR, 'material.pkl')  # doc_id  -> [word1, word2, ...], list
weighted_embedding_path = './output/weighted_embedding.pkl'  # doc_id  -> X_i, np.ndarray
rule_check_file_path    = join(OUTPUT_DIR, 'rule_check.json')
pos_pair_path           = join(OUTPUT_DIR, 'pos_pair.json')
single_dogs_path        = join(OUTPUT_DIR, 'single_dogs.json')
#word2vect_model_path    = join(DATA_DIR, 'GoogleNews-vectors-negative300.bin.gz')  # word2vec model.  usage: KeyedVectors.load_word2vec_format(...)
word2vect_model_path    = './output/word.emb'                   # word2vec model.  usage: Word2Vec.load(...)
triple_set              = './output/triple.pkl'                           # 'emb'   -> anchors; 'emb_pos': positive weighted embedding; 'emb_neg': negative ones
EMBEDDING_DIM = 100

CPU_COUNT = 20
