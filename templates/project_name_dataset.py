import pickle

from torch.utils.data import dataset
from torchvision import transforms

TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ProjectNameDataset(dataset.Dataset):
    def __init__(self, config, mode):
        super(ProjectNameDataset, self).__init__()
        self.config = config
        self.mode = mode
        with open(config.IMDB_PATH, 'rb') as f:
            self.imdb = pickle.load(f)[mode]

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError
