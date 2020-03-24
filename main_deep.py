import os
import time
import numpy as np
import torch

from utils import *
from visualisation import *
from time_frequence import *
from feature_extraction import *
from eeg_nn import *

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import random

random.seed(42)
np.random.seed(seed=42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ =="__main__":

    start = time.time()
    print('Loading dataset...')
    X_train = load_h5(os.getcwd() + '/X_train_new.h5')
    X_test = load_h5(os.getcwd() + '/X_test_new.h5')
    train_idx, train_labels = load_csv(os.getcwd() + '/y_train_AvCsavx.csv')
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    

    # Balancing the datasets
    print('Balancing datset...')
    X_train, train_labels = balance(X_train, train_labels)
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    print('Re-sampling for data augmentation...')
    re_sample_rate = 2
    X_train = re_sample(X_train, re_sample_rate)
    #train_labels = make_outputs(train_labels, n_tests=re_sample_rate)
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    '''print('Scaling dataset...')
    X_train = scale_channels(X_train, custom=True)
    print('Done in {}'.format(time.time()-start))
    start = time.time()'''

    a,b,c,d = X_train.shape
    X_train = np.reshape(X_train, (a*b,c,d))
    train_labels = make_outputs(train_labels, b)

    val_feats, y_val = X_train[-100*b:], np.array(train_labels[-100*b:])
    train_feats, y_train = X_train[:-100*b], np.array(train_labels[:-100*b])
    print(train_feats.shape, y_train.shape)

    use_cuda = torch.cuda.is_available()
    if use_cuda: print('Using CUDA!')

    datasetTrain = dataset(train_feats.astype("float32"),y_train.astype("float32"), cuda=use_cuda)
    datasetVal = dataset(val_feats.astype("float32"),y_val.astype("float32"), cuda=use_cuda)

    model = Egg_module(lr=0.0001, cuda=use_cuda, n_samples=val_feats.shape[2])
    train_losses, eval_losses = model.train(datasetTrain, batch_size=64, epochs=200, shuffle = True, test = datasetVal)

    plt.figure()
    plt.plot(train_losses, color='b', label='train')
    plt.plot(eval_losses, color='g', label='eval')
    plt.legend()
    plt.show()
