from .code import CodeUtil
from .mol import MolUtil
from .tud import TUUtil

DATASET_UTILS = {
    'ogbg-code': CodeUtil,
    'ogbg-code2': CodeUtil,
    'ogbg-molhiv': MolUtil,
    'ogbg-molpcba': MolUtil,
    'ogbg-tox21': MolUtil,
    'ogbg-toxcast': MolUtil,
    'NCI1': TUUtil,
    'NCI109': TUUtil,
    'MUTAG': TUUtil,
    'ENZYMES': TUUtil,
    'DD': TUUtil,
    'PROTEINS': TUUtil,
    'IMDB-BINARY': TUUtil,
    'IMDB-MULTI': TUUtil,
    'PTC_MR': TUUtil,
    'Mutagenicity': TUUtil,
    'FRANKENSTEIN': TUUtil,
    'REDDIT_BINARY': TUUtil,
    'COLLAB': TUUtil
}
