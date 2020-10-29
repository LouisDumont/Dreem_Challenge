import os
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import torch

from utils import *
from visualisation import *
from time_frequence import *
from eeg_nn import *

random.seed(42)
np.random.seed(seed=42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train():
    start = time.time()
    print('Loading dataset...')
    X_train = load_h5(os.getcwd() + '/X_train_new.h5')
    X_test = load_h5(os.getcwd() + '/X_test_new.h5')
    train_idx, train_labels = load_csv(os.getcwd() + '/y_train_AvCsavx.csv')
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    val_feats, y_val = X_train[-150:], np.array(train_labels[-150:])
    train_feats, y_train = X_train[:-150], np.array(train_labels[:-150])
    print(train_feats.shape, y_train.shape)

    balance_set = True

    # Balancing the datasets
    print('Balancing datset...')
    val_feats, y_val = balance(val_feats, y_val)
    if balance_set:
        train_feats, y_train = balance(train_feats, y_train)
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    print('Re-sampling for data augmentation...')
    re_sample_rate = 2
    val_feats = re_sample(val_feats, re_sample_rate)
    train_feats = re_sample(train_feats, re_sample_rate)
    # train_labels = make_outputs(train_labels, n_tests=re_sample_rate)
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    '''print('Scaling dataset...')
    X_train = scale_channels(X_train, custom=True)
    print('Done in {}'.format(time.time()-start))
    start = time.time()'''

    a, b, c, d = val_feats.shape
    val_feats = np.reshape(val_feats, (a*b, c, d))
    y_val = np.array(make_outputs(y_val, b))
    a, b, c, d = train_feats.shape
    print("training set shape:", train_feats.shape)
    train_feats = np.reshape(train_feats, (a*b, c, d))
    y_train = np.array(make_outputs(y_train, b))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA!')

    weights = None
    if not balance_set:
        s1 = np.sum(y_train)
        s0 = np.sum(1-y_train)
        print(s0)
        print(s1)
        x0 = s0/(s0+s1)
        x1 = s1/(s0+s1)
        reg = 1/(2*x0*x1)
        weights = torch.from_numpy(np.array([reg*x1, reg*x0])).float()
        if use_cuda:
            weights = weights.cuda()
    print('weights:', weights)

    datasetTrain = dataset(train_feats.astype("float32"), y_train.astype("float32"), cuda=use_cuda)
    datasetVal = dataset(val_feats.astype("float32"), y_val.astype("float32"), cuda=use_cuda)

    model = Egg_module(lr=0.0002, cuda=use_cuda, n_samples=val_feats.shape[2],
                       criterion=torch.nn.CrossEntropyLoss(weight=weights))

    pytorch_total_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print('Number of trainable parameters:', pytorch_total_params)

    train_losses, eval_losses = model.train(datasetTrain, batch_size=64, epochs=400,
                                            shuffle=True, test=datasetVal)

    model.save_weight()

    plt.figure()
    plt.plot(train_losses, color='b', label='train')
    plt.plot(eval_losses, color='g', label='eval')
    plt.legend()
    plt.show()


def predict():
    start = time.time()
    print('Loading dataset...')
    X_test = load_h5(os.getcwd() + '/X_test_new.h5')
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    print('Re-sampling for data augmentation...')
    re_sample_rate = 2
    test_feats = re_sample(X_test, re_sample_rate)
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    a, b, c, d = test_feats.shape
    test_feats = np.reshape(test_feats, (a*b,c,d))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA!')

    model = Egg_module(lr=0.0002, cuda=use_cuda, n_samples=test_feats.shape[2],
                       criterion=torch.nn.CrossEntropyLoss())
    model.load()

    raw_res = []
    for i in range(a*b):
        in_feat = torch.from_numpy(test_feats[i]).view(1, c, d).float()
        if use_cuda:
            in_feat = in_feat.cuda()
        # print(in_feat.shape)
        out = model.predict(in_feat)
        raw_res.append(np.argmax(out))
    indiv_preds = make_indiv_pred(raw_res, b)

    write_preds(indiv_preds)


if __name__ =="__main__":

    train()
    # predict()
