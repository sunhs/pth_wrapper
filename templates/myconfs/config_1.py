import os
import pickle

######################################################################
# Needed by my_modules.
######################################################################
# Names and paths.
MODEL_NAME = 'EAST'
PROJ_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT_DIR, 'data')
PRETRAIN_PATH = ''
MODEL_DIR = os.path.join(
    PROJ_ROOT_DIR, 'save_dir', 'model_{}'.format(
        os.path.basename(__file__).rsplit('.')[0].split('_')[1]))
STATE_DIR = os.path.join(MODEL_DIR, 'states')
IMG_ROOT_DIR = '/root/share/dataset/icdar15_it'
IMDB_PATH = os.path.join(DATA_DIR, 'imdb.pkl')
STATE_PREFIX = 'epoch'

# Hypers and devices.
MAX_EPOCHS = 2400
BATCH_SIZE = {'train': 32, 'test': 32}
PARAM_GROUPS = [{
    'params': ['default'],
    'lr': 1e-4,
    'weight_decay': 1e-5
}, {
    'params': ['score_tail'],
    'lr': 1e-3,
    'weight_decay': 1e-5
}]
GPUS = []
DEFAULT_GPU = 2
NUM_WORKERS = 8

# Optional.
SAVE_EPOCH_FREQ = 40
# uncomment these lines to send logs to your email
# EMAIL = True
# EMAIL_ADDR = 'Your Email Address'

if not os.path.exists(STATE_DIR):
    os.makedirs(STATE_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

with open(os.path.join(MODEL_DIR, 'log.pkl'), 'wb') as f:
    log = {'train': {'loss': []}}
    pickle.dump(log, f)

with open(os.path.join(MODEL_DIR, 'train_log.pkl'), 'wb') as f:
    train_log = []
    pickle.dump(train_log, f)
