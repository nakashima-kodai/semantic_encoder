import os
import matplotlib.pyplot as plt
import torch
import torchvision


def show_loaded_image(image_A, image_B, nrow):
    grid_image = torch.cat((image_A, image_B), dim=0)
    grid_img = torchvision.utils.make_grid(grid_image, nrow=nrow, padding=1, normalize=False)
    grid_img = grid_img.numpy().transpose((1, 2, 0))

    plt.imshow(grid_img)
    plt.pause(0.5)
    plt.clf()
    # plt.show()

def save_image(opt, epoch, saved_image):
    saved_image = torchvision.utils.make_grid(saved_image.cpu(), nrow=2*opt.batch_size, padding=1, normalize=True)
    saved_image = saved_image.numpy().transpose((1, 2, 0))

    image_name = str(epoch).zfill(3) + '.png'
    save_path = os.path.join(opt.sample_dir, opt.name, image_name)
    print('save_path: {}\n'.format(save_path))
    plt.imsave(save_path, saved_image)

def print_current_losses(epoch, iter, losses):
    message = '---------------------------------------------------\n'
    message += '           epoch : {:03d}, iters : {:03d}\n'.format(epoch, iter)
    message += '---------------------------------------------------\n'
    for k, v in losses.items():
        message += ('{:>11} : {:.7f}\n'.format(k, v))

    print(message)

def print_epoch_losses(epoch, iter, losses):
    message = '---------------------------------------------------\n'
    message += '                   epoch : {:03d}\n'.format(epoch)
    message += '---------------------------------------------------\n'
    for k, v in losses.items():
        message += ('{:>11} : {:.7f}\n'.format(k, v/(iter+1)))

    print(message)
