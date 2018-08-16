import pickle

from torch.utils.data import dataset
from torchvision import transforms

TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Dataset(dataset.Dataset):
    def __init__(self, config, mode):
        """Initializes the dataset instance. Loads necessary data such as imdb.

        :param config: The global config.
        :param mode: Specifies the mode, either 'train' or 'test'. (string)
        :returns: None
        :rtype: None

        """
        super(Dataset, self).__init__()
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
        """Define this function if the default collate function defined in
        torch.utils.data.dataloader doesn't meet your needs. Refer to the
        default one to find out how it works.

        :param batch: A batch of data. (list)
        :returns: The collated batch of data.

        """
        raise NotImplementedError
