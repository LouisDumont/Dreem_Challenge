import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def show_ts(array_4d, idx=(0, 1), title=''):
    '''
    Plots and shows 4 channels on each class
    Parameters
    ----------
    array_4d: numpy array
        array representing the input data
    idx: tuple of int
        represent the indexes to plot in the first dimention of the array
        Ideally, one index should represent each class
    '''
    # PLot 4 channels on a 0 subject
    fig = plt.figure()
    plt.title(title + 'first sample')
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(array_4d[idx[0], 0, i])

    # PLot 4 channels on a 1 subject
    fig = plt.figure()
    plt.title(title + 'second sample')
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(array_4d[idx[1], 0, i])

    plt.show()


def show_PCA(array_2d, labels):
    '''
    Plot the first 2 dimensions of the PCA, with colors depending on labels
    '''
    labels = np.array(labels)
    model = PCA(n_components=2)
    reduced_X = model.fit_transform(array_2d)

    plt.figure()
    plt.scatter(reduced_X[labels == 0][:, 0], reduced_X[labels == 0][:, 1], color='r')
    plt.scatter(reduced_X[labels == 1][:, 0], reduced_X[labels == 1][:, 1], color='b')

    plt.show()
