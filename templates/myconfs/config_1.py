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
SAVE_EPOCH_FREQ = 40
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
MILESTONES = [800, 1600]
# uncomment these lines to send logs to your email
# EMAIL = True
# EMAIL_ADDR = 'Your Email Address'

######################################################################
# East
######################################################################
TEST_SIZE = (704, 1280)  # (H, W)
BACKBONE = 'resnet50'
MIN_CROP_SIDE_RATIO = 0.1
BACKGROUND_RATIO = 3.0 / 8
RANDOM_SCALE = [0.5, 1, 2, 3]
TRAIN_INPUT_SIZE = 512
MIN_TEXT_SIZE = 8

######################################################################
# Deepcoloring
######################################################################
MAX_COLOR = 9
MAX_INS = 30
HALO_MARGIN = 21
K_NEG = 7

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
