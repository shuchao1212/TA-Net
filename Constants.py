import os
import shutil

Image_size = (448, 448)
#ROOT = './dataset/Colon_p'
#ROOT = './dataset/Liver'
ROOT = './dataset/DRIVE'
BATCHSIZE_PER_CARD = 4
TOTAL_EPOCH = 300
INITAL_EPOCH_LOSS = 10000
NUM_EARLY_STOP = 20
NUM_UPDATE_LR = 10

BINARY_CLASS = 1
