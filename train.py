import os
import torch
from tensorboardX import SummaryWriter
from options.train_options import TrainOptions
from data.create_data_loader import create_dataloader
from models import create_model
from utils import visualizer


opt = TrainOptions().parse()

ckpt_dir = os.path.join(opt.ckpt_dir, opt.name)
writer = SummaryWriter(ckpt_dir)

# set dataloader
print('### prepare DataLoader')
source_loader = create_dataloader(opt, is_source=True)
target_loader = create_dataloader(opt, is_source=False)
print('source images = {}'.format(len(source_loader)*opt.batch_size))
print('target images = {}'.format(len(target_loader)*opt.batch_size))

# set model
model = create_model(opt)

for epoch in range(opt.epoch+opt.epoch_decay+1):
    for iter, (src_data, tgt_data) in enumerate(zip(source_loader, target_loader)):
        model.set_variables(src_data, tgt_data)
        model.optimize_parameters()
        model.sum_epoch_losses()

        if iter % opt.print_iter_freq == 0:
            losses = model.get_current_losses()
            visualizer.print_current_losses(epoch, iter, losses)

            saved_image = model.forward()
            visualizer.save_image(opt, epoch, saved_image)

    if epoch % opt.save_epoch_freq == 0:
        saved_image = model.forward()
        visualizer.save_image(opt, epoch, saved_image)

        model.save_networks(epoch)

    model.update_lr()

    for k, v in model.epoch_losses.items():
        writer.add_scalar(k, v/(iter+1), epoch)

    visualizer.print_epoch_losses(epoch, iter, model.epoch_losses)
    model.reset_epoch_losses()
