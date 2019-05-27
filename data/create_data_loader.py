from torch.utils import data
from .bdd100k_dataset import BDD100KDataset
from .gta5_dataset import GTA5Dataset


def create_dataloader(opt, is_source=True):
    dataset_name = opt.source_dataset if is_source else opt.target_dataset

    if dataset_name == 'bdd100k':
        dataset = BDD100KDataset(opt, is_source)
    elif dataset_name == 'gta5':
        dataset = GTA5Dataset(opt, is_source)
    else:
        raise ValueError('Dataset [%s] not recognized.' % (dataset_name))

    dataloader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.no_shuffle, drop_last=True)
    print('[%s] loader was created' % (dataset.name()))
    return dataloader
