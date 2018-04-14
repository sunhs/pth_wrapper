import os

MODEL_NAME = ''
PROJ_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT_DIR, 'data')
PRETRAIN_PATH = ''
MODEL_DIR = os.path.join(PROJ_ROOT_DIR, 'model_dir', 'model_{}'.format(
    os.path.basename(__file__).rsplit('.')[0].split('_')[1]))
STATE_DIR = os.path.join(MODEL_DIR, 'states')
IMG_ROOT_DIR = ''
IMDB_PATH = os.path.join(DATA_DIR, 'imdb.pickle')
STATE_PREFIX = 'epoch'
NUM_CLASSES = None
BATCH_SIZE = {'train': None, 'test': None}
MAX_EPOCHS = None
GPUS = []
DEFAULT_GPU = 0
PARAM_GROUPS = [{'params': ['default'], 'lr': 1e-4, 'weight_decay': 5e-4}]

# uncomment these lines to send logs to your email
# EMAIL = True
# EMAIL_ADDR = 'Your Email Address'

if not os.path.exists(STATE_DIR):
    os.makedirs(STATE_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
