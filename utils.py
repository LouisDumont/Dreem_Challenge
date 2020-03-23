import os
import time
import csv
import h5py
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing


def load_h5(path):
    '''
    h5 to np array. Expects list of 3D array in h5
    '''
    with h5py.File(path, 'r') as file:
        # List all groups
        a_group_key = list(file.keys())[0]

        # Get the data
        data = (file[a_group_key]).value
        n_tests, n_channels, n_samples = data[0].shape
        
        res = np.zeros((len(data), n_tests, n_channels, n_samples))
        for i, item in enumerate(data):
            res[i] = item
    return res

def load_csv(path):
    '''
    csv to list, expects 2 columns and first row of labels
    '''
    idx, classes = [], []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0]!='id':
                idx.append(int(row[0]))
                classes.append(int(row[1]))
    return idx, classes

def make_inputs(tab):
    '''
    Resizes the input array so that it can be fed into most classifiers
    '''
    n_patients, n_tests, n_features = tab.shape
    return np.resize(tab, (n_patients*n_tests, n_features))

def make_outputs(out, n_tests=40):
    '''
    Extends output list so that it corresponds to final input tab
    '''
    res = []
    for val in out:
        for i in range(n_tests):
            res.append(val)
    return(res)

def make_indiv_pred(preds, n_tests=40):
    '''
    Makes a single prediction per patient out of test-wise predictions
    '''
    n_preds = len(preds)
    assert n_preds%n_tests==0
    n_patients = n_preds//n_tests
    final_preds = []
    for patient in range(n_patients):
        sum_preds = 0
        for i in range(n_tests):
            sum_preds += preds[patient*n_tests + i]
        final_preds.append(sum_preds/n_tests >= 0.5)
    return final_preds

def balance(X_train, train_labels):
    '''
    Selects samples from the dataset to balance it (attention: may throw away most of the dataset)
    '''
    train_labels = np.array(train_labels)
    Xtrain_pos = X_train[train_labels==1]
    # Select as many zero samples as there are positive samples
    Xtrain_zero = X_train[train_labels==0][:Xtrain_pos.shape[0]]
    nXtrain = np.concatenate([Xtrain_pos, Xtrain_zero], axis=0)
    nYtrain = [1 for i in range(Xtrain_pos.shape[0])] + [0 for i in range(Xtrain_zero.shape[0])]
    nXtrain, nYtrain = shuffle(nXtrain, nYtrain)
    return (nXtrain, nYtrain)

def create_features_list(tab, features):
    '''
    From the original samples and a list of features functions, returns the array of features.
    '''
    n_patients, n_tests, n_channels, n_samples = tab.shape
    n_feats = len(features)
    res = np.zeros((n_patients, n_tests, n_feats*n_channels))
    for patient in range(n_patients):
        for test in range(n_tests):
            for channel in range(n_channels):
                for idx, feature in enumerate(features):
                    #print(feature(tab[patient, test, channel]))
                    res[patient, test, channel*n_feats + idx] = feature(tab[patient, test, channel])
    return res

def create_features_func(tab, function):
    '''
    From the original samples and a function to apply to each channel, returns the array of features.
    '''
    n_patients, n_tests, n_channels, n_samples = tab.shape
    res_len = function(tab[0,0,0]).shape[0]
    res = np.zeros((n_patients, n_tests, res_len*n_channels))
    for patient in range(n_patients):
        for test in range(n_tests):
            res_list = []
            for channel in range(n_channels):
                res_list.append(function(tab[patient, test, channel]))
            res[patient, test] = np.concatenate(res_list, axis=0)
    return res

def map_channels(tab, transform):
    '''
    Applies transform to each channel of the tab
    '''
    n_patients, n_tests, n_channels, n_samples = tab.shape
    res_len = transform(tab[0,0,0]).shape[0]
    res = np.zeros((n_patients, n_tests, n_channels, res_len))
    for patient in range(n_patients):
        for test in range(n_tests):
            for channel in range(n_channels):
                res[patient, test, channel] = transform(tab[patient, test, channel])
    return res

def re_sample(tab, n_mult):
    '''
    Multiply the number of samples in one dataset by dividing the time series in n_mult sub-time series
    '''
    n_patients, n_samples, n_channels, n_feat = tab.shape
    print(n_feat, n_mult)
    assert n_feat % n_mult == 0
    new_l = n_feat//n_mult
    res = np.zeros((n_patients, n_samples*n_mult, n_channels, n_feat//n_mult))
    for patient in range(n_patients):
        for sample in range(n_samples):
            for i in range(n_mult):
                for channel in range(n_channels):
                    res[patient, sample*n_mult+i, channel] = tab[patient, sample, channel, i*new_l:(i+1)*new_l]
    return res

def scale_channels(tab, custom=True):
    '''
    Each channel with be standardized (accross all samples) to have mean=0 and std=1
    '''
    n_patients, n_tests, n_channels, n_feat = tab.shape
    res = np.zeros(tab.shape)
    for i in range(n_channels):
        if custom:
            x = tab[:,:,i,:]
            res[:,:,i,:] = (x-np.mean(x))/np.std(x)
        else:
            res[:,:,i,:] = preprocessing.scale(tab[:,:,i,:])
    return res

def scale_features(tab, custom=True):
    '''
    Each channel with be standardized (accross all samples) to have mean=0 and std=1
    '''
    n_patients, n_tests, n_features = tab.shape
    res = np.zeros(tab.shape)
    for i in range(n_features):
        if custom:
            x = tab[:,:,i]
            res[:,:,i] = (x-np.mean(x))/np.std(x)
        else:
            res[:,:,i] = preprocessing.scale(tab[:,:,i])
    return res
