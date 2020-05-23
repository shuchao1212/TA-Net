# TA-Net
TA-Net: Better Feature Selection, Better Tumor Segmentation

(This work was submitted to Neural Networks, and now it is revising.)
# Introduction about this Repository
1. dataset folder: place your datasets. Here, we give a example from DRIVE dataset used in our paper, please refer to relevant papers.
2. intermediate_results folder: observe some outputs after every epoch when training our networks.
3. logs folder: record some training information after every epoch during the period of training.
4. networks folder: place your network model. Here, tanet.py is our model used in our submitted paper.
5. test_results folder: save outputs when testing the well-trained model on test set.
6. weights folder: save the best model and its parameters during the period of training.
7. Constants.py: set some hyper-parameters, e.g., epochs, which dataset used, classes, etc.
8. data.py: generate dataloader for training, including how to read images and labels for different datasets.
9. framework.py: code some information about training networks.
10. loss.py: set the loss candidates.
11. test_tanet.py: test the trained model on test set.
12. train_tanet.py: train our network model on training set.

# Used Python and Pytorch Versions
Python 3.7.4, Pytorch 1.3.1, one GPU
# How to build your dataset
You can refer to the given example, DRIVE, to set your own data.
# How to train it
You can directly run train_tanet.py if your have one GPU resource. It will start to train our TA-Net on the training set of DRIVE dataset. Then, you will observe some outputs of training data in intermediate_results folder after every epoch. After runing 300 epochs we set in Constants.py, you can use the trained network parameters in weights folder to test the test set of DRIVE dataset. (Similarly, you can replace DRIVE dataset with yours to implement these steps. So easy!)
# How to test it
After the model is well trained, you can directly run test_tanet.py to obtain test results which will save in test_results folder.
# Please refer to our paper if you used our codes, thanks.
Shuchao Pang, Anan Du, Mehmet A. Orgun, Yunyun Wang, and Zhenmei Yu, “TA-Net: Better Feature
Selection, Better Tumor Segmentation”. Neural Networks. (Under Review -> Revising)
