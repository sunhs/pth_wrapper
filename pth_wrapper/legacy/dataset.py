from multiprocessing import Pool
import pickle
import random
import threading


class Dataset:

    def __init__(self, config):
        self.config = config
        self.pool = Pool()

        if config.MEAN_PATH:
            with open(config.MEAN_PATH, 'rb') as f:
                self.mean = pickle.load(f).astype('float32')
        with open(config.IMDB_PATH, 'rb') as f:
            self.imdb = pickle.load(f)

        self.ds_size = {}
        self.dataset = {}
        self._cur_index = {}
        self.data = {}
        self.update_thread = {}
        self.images = {}

        for mode in ('train', 'test'):
            self.ds_size[mode] = self.imdb[mode]['names'].size
            self.dataset[mode] = list(range(self.ds_size[mode]))
            self._cur_index[mode] = 0
            self.data[mode] = None
            self.update_thread[mode] = None

    def shuffle_train_set(self):
        random.shuffle(self.dataset['train'])

    def next_batch(self, batch_size, mode):
        if self.data[mode] is None:
            self.update_thread[mode] = threading.Thread(
                target=self.update, args=(batch_size, mode))
            self.update_thread[mode].start()

        self.update_thread[mode].join()
        data = self.data[mode]
        self.update_thread[mode] = threading.Thread(
            target=self.update, args=(batch_size, mode))
        self.update_thread[mode].start()
        return data

    def update(self, batch_size, mode):
        end = min(self._cur_index[mode] + batch_size, self.ds_size[mode])
        batch = self.dataset[mode][self._cur_index[mode]: end]
        self._cur_index[mode] = end if end < self.ds_size[mode] else 0
        self.data[mode] = self._get_batch(batch, mode)

    def _get_batch(self, batch, mode):
        raise NotImplementedError
