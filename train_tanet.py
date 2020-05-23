from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

from networks.tanet import TA_Net_
from framework import MyFrame
from loss import dice_bce_loss
#from loss import dice_loss
from data import ImageFolder
#from Visualizer import Visualizer

import Constants
#import image_utils
savepath='intermediate_results/'
import numpy as np

# Please specify the ID of graphics cards that you want to use
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


def TA_Net_Train():
    NAME = 'TA-Net' + Constants.ROOT.split('/')[-1]

    # run the Visdom
    #viz = Visualizer(env=NAME)

    solver = MyFrame(TA_Net_, dice_bce_loss, 2e-4)
    #batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD
    batchsize = 4
    # print the total number of parameters in the network
    #solver.paraNum()

    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.

    dataset = ImageFolder(root_path=Constants.ROOT, datasets='DRIVE')#'Colon_p'
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)

    # print the total number of parameters in the network
    solver.paraNum() 
    
    # load model
    #solver.load('./weights/' + NAME + '_plus_spatial_multi.th')

    # start the logging files
    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()

    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0

        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss
            index = index + 1

        print('epoch:', epoch, '    time before imwrite:', int(time() - tic))

        # show the original images, predication and ground truth on the visdom.
        show_image = (img + 1.6) / 3.2 * 255.
        #viz.img(name='images', img_=show_image[0, :, :, :])
        #viz.img(name='labels', img_=mask[0, :, :, :])
        #viz.img(name='prediction', img_=pred[0, :, :, :])
        cv2.imwrite(savepath+'img0-tanet-'+str(epoch)+'.png',np.transpose(show_image[0,:,:,:].cpu().detach().numpy(),(1,2,0)))
        cv2.imwrite(savepath+'mask0-tanet-'+str(epoch)+'.png',np.transpose(mask[0,:,:,:].cpu().detach().numpy()*255,(1,2,0)))
        cv2.imwrite(savepath+'pred0-tanet-'+str(epoch)+'.png',np.transpose(pred[0,:,:,:].cpu().detach().numpy()*255,(1,2,0)))
        

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        print(mylog, '********')
        print(mylog, 'epoch:', epoch, '    time:', int(time() - tic))
        print(mylog, 'train_loss:', train_epoch_loss)
        print(mylog, 'SHAPE:', Constants.Image_size)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('totalNum in an epoch:',index)
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', Constants.Image_size)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + NAME + '_plus_spatial_multi.th')
        if no_optim > Constants.NUM_EARLY_STOP:
            print(mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + NAME + '_plus_spatial_multi.th')
            solver.update_lr(2.0, factor=True, mylog=mylog)
        mylog.flush()

    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    TA_Net_Train()



