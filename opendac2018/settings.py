from os.path import abspath, dirname, join
import os

PROJ_DIR = join(abspath(dirname(__file__)), '.')
DATA_DIR = join(PROJ_DIR, './data/')
OUTPUT_DIR = join(PROJ_DIR, './output/')
assignments_train_path = join(DATA_DIR, 'assignment_train.json')
pubs_validate_path = join(DATA_DIR, 'pubs_validate.json')
TRAIN_NAME2PUB = join(DATA_DIR, 'assignment_train.json')
VAL_NAME2PUB = join(DATA_DIR, 'await_validation.json')
VAL_PATH = join(DATA_DIR, 'pubs_validate.json')

idf_path = join(OUTPUT_DIR, 'idf.pkl')  # word    -> idf value, float
global_output_path = join(OUTPUT_DIR,
                          'global_output.pkl')  # doc_id  -> Y_i, np.ndarray
local_output_path = join(OUTPUT_DIR,
                         'local_output.pkl')  # doc_id  -> Z_i, np.ndarray
material_path = join(OUTPUT_DIR,
                     'material.pkl')  # doc_id  -> [word1, word2, ...], list
IDF_THRESHOLD = 32
