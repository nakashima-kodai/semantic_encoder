import torch
from options.train_options import TrainOptions
from data.create_data_loader import create_source_dataloader, create_target_dataloader
from utils import visualizer


opt = TrainOptions().parse()

# set dataloader
print('### prepare DataLoader')
source_loader = create_source_dataloader(opt)
target_loader = create_target_dataloader(opt)
print('source images = {}'.format(len(source_loader)*opt.batch_size))
print('target images = {}'.format(len(target_loader)*opt.batch_size))

for iter, (src_data, tgt_data) in enumerate(zip(source_loader, target_loader)):
    # src_img = (src_data['image'] + 1.0) / 2.0
    # tgt_img = (tgt_data['image'] + 1.0) / 2.0
    # src_lbl = src_data['label'].repeat(1, 3, 1, 1)
    # tgt_lbl = tgt_data['label'].repeat(1, 3, 1, 1)
    # src_imgs = torch.cat((src_img, src_lbl), dim=0)
    # tgt_imgs = torch.cat((tgt_img, tgt_lbl), dim=0)
    #
    # visualizer.show_loaded_image(src_imgs, tgt_imgs, nrow=opt.batch_size)

    print(iter)
