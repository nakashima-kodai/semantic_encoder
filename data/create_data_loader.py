from torch.utils import data
from .bdd100k_dataset import BDD100KDataset


def create_source_dataloader(opt):
    if opt.source_dataset == 'bdd100k':
        source_dataset = BDD100KDataset(opt, source=True)
    else:
        raise ValueError('Source Dataset [%s] not recognized.' % (opt.source_dataset))

    souce_dataloader = data.DataLoader(source_dataset, batch_size=opt.batch_size, shuffle=not opt.no_shuffle, drop_last=True)
    print('[%s] loader was created' % (source_dataset.name()))
    return souce_dataloader

def create_target_dataloader(opt):
    if opt.target_dataset == 'bdd100k':
        target_dataset = BDD100KDataset(opt, source=False)
    else:
        raise ValueError('Target Dataset [%s] not recognized.' % (opt.taret_dataset))

    target_dataloader = data.DataLoader(target_dataset, batch_size=opt.batch_size, shuffle=not opt.no_shuffle, drop_last=True)
    print('[%s] loader was created' % (target_dataset.name()))
    return target_dataloader
