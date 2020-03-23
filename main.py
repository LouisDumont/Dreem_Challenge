import os
import time
import numpy as np

from utils import *
from visualisation import *
from time_frequence import *
from feature_extraction import *

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import random

random.seed(42)
np.random.seed(seed=42)

if __name__ =="__main__":

    start = time.time()
    print('Loading dataset...')
    X_train = load_h5(os.getcwd() + '/X_train_new.h5')
    X_test = load_h5(os.getcwd() + '/X_test_new.h5')
    train_idx, train_labels = load_csv(os.getcwd() + '/y_train_AvCsavx.csv')
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    #show_ts(X_train)

    #show_ts(ts_train) # That function is not adapted

    # Balancing the datasets
    print('Balancing datset...')
    X_train, train_labels = balance(X_train, train_labels)
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    # Re-sample
    '''print('Re-sampling for data augmentation...')
    re_sample_rate = 4
    X_train = re_sample(X_train, re_sample_rate)
    #train_labels = make_outputs(train_labels, n_tests=re_sample_rate)
    print('Done in {}'.format(time.time()-start))
    start = time.time()'''

    # Scale all channels
    '''print('Scaling all channels')
    X_train = scale_channels(X_train)
    print('Done in {}'.format(time.time()-start))
    start = time.time()'''

    print('Computing Welch transforms of the inputs...')
    X_train = map_channels(X_train, lambda x: own_welch(x, 50)) # Attention! the Welch dim do not correspond to frequency!!
    # !! Frequencies change is nperseg changes (unclear doc)
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    '''for i in range(X_train.shape[2]):
        print(np.mean(X_train[:,:,i,:]), np.std(X_train[:,:,i,:]))'''

    # Create the formated inputs
    # Split
    print('Spliting and formating sets...')
    features_list = [lambda x: x[0], lambda x: x[1], lambda x: x[2], lambda x: x[3], lambda x: x[4], lambda x: x[5], lambda x: x[6]]
    # [(lambda x: extract_index(x, i)) for i in range(7)]
    # [np.amax, np.mean, np.min, np.std]
    X_train = create_features_list(X_train, features_list)

    print('Scaling all features')
    X_train = scale_features(X_train)
    print('Done in {}'.format(time.time()-start))

    start = time.time()
    val_feats, y_val = X_train[-100:], train_labels[-100:]
    train_feats, y_train = X_train[:-100], train_labels[:-100]
    # Extract formal features
    '''train_feats = create_features_list(x_train, features_list) # [(lambda x: extract_index(x, i)) for i in range(7)]
    val_feats = create_features_list(x_val, features_list) # [np.amax, np.mean, np.min, np.std]'''

    print('train_feats shape:', train_feats.shape)
    '''for i in range(train_feats.shape[2]):
        print(np.mean(train_feats[:,:,i]), np.std(train_feats[:,:,i]))'''

    # Formalize the dataset (n_samples*n_features)
    in_train, in_val = make_inputs(train_feats), make_inputs(val_feats)
    n_samples = X_train.shape[1]

    out_train, out_val = make_outputs(y_train, n_tests=n_samples), make_outputs(y_val, n_tests=n_samples)
    print('in_train shape:', in_train.shape)
    print('in_val shape:', in_val.shape)
    print('Balance in train:', sum(y_train)/len(y_train))
    print('Balance in val:', sum(y_val)/len(y_val))
    # Make a single prediction per patient
    '''print(train_feats.shape)
    nx, ny, nz = train_feats.shape
    in_train, in_val = np.reshape(train_feats, (nx, ny*nz)), np.reshape(val_feats, (-1, ny*nz))
    out_train, out_val = y_train, y_val'''
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    # Do and print PCA to debug
    '''print(in_train.shape)
    for i in range(in_train.shape[1]):
        print(np.mean(in_train[:,i]), np.std(in_train[:,i]))'''
    show_PCA(in_train, out_train)

    pca_model = PCA(n_components=2)
    in_train = pca_model.fit_transform(in_train)
    in_val = pca_model.transform(in_val)

    print('Training and testing model...')
    print('Final shape of the training set:', in_train.shape)
    #model = LogisticRegression() # class_weight='balanced'
    model = SVC()
    model.fit(in_train, out_train)
    print('Accuracy on train:', model.score(in_train, out_train))
    print('Accuracy on val:', model.score(in_val, out_val))
    print('Proportion of 1 predicted:', np.sum(model.predict(in_val))/len(out_val))
    print('Done in {}'.format(time.time()-start))
    start = time.time()

    # Print final accuracy:
    finpred_train = make_indiv_pred(model.predict(in_train), n_tests=n_samples)
    finpred_val = make_indiv_pred(model.predict(in_val), n_tests=n_samples)
    print('Accuracy on final prediction (train):', accuracy_score(y_train, finpred_train))
    print('Accuracy on final prediction (val):', accuracy_score(y_val, finpred_val))
    print('Proportion of 1 predicted(val):', sum(finpred_val)/len(finpred_val))
