"""
.. module:: DataGenerators

DataGenerators
*************

:Description: DataGenerators

    

:Authors: shreyas
    

:Version: 

:Created on: 21/12/2016 8:13 

"""

__author__ = 'shreyas'

from numpy.random import shuffle
import numpy as np
from keras.utils import np_utils


def list_days_generator(year, month, iday, fday):
    """
    Generates a list of days

    :param year:
    :param month:
    :param iday:
    :param fday:
    :return:
    """
    ldays = []
    for v in range(iday, fday+1):
        ldays.append("%d%d%02d" % (year, month, v))
    return ldays


def load_days(dataset_path, days, z_factor, reb=False):
    """
    loads and contatenates files from a list of days
    :param days:
    :param z_factor:
    :param reb:
    :return:
    """
    ldata = []
    labels = []
    fnamed = 'data'
    fnamel = 'labels'
    if reb:
        fnamed = 'r' + fnamed
        fnamel = 'r' + fnamel
    for day in days:
        data = np.load(dataset_path + fnamed + '-D%s-Z%0.2f.npy' % (day, z_factor))
        ldata.append(data)
        labels.extend(np.load(dataset_path + fnamel + '-D%s-Z%0.2f.npy' % (day, z_factor)))
    data = np.concatenate(ldata)
    return data, labels


def simpleDataGenerator(datapath, days, z_factor, nclasses, batchsize, groups):
    """
    Loops through the day files yielding a batch of examples
    Files are loaded in groups and batches are randomized
    :param datapath
    :param days:
    :param z_factor:
    :param nclasses:
    :param batchsize:
    :param groups:
    :return:
    """
    while True:
        shuffle(days)
        lgroups = []
        for i in range(0, len(days), groups):
            group = []
            for j in range(groups):
                if (i + j) < len(days):
                    group.append(days[i + j])
            lgroups.append(group)

        for lday in lgroups:
            data, labels = load_days(datapath,lday, z_factor)

            limit = (data.shape[0]//batchsize) - 1
            X_train = data
            # Data generated in Theano order
            # if imgord == 'th':
            #     X_train = data.transpose((0,3,1,2))
            # else:
            #     X_train = data

            y_trainO = labels
            y_train = np_utils.to_categorical(y_trainO, nclasses)
            perm = range(X_train.shape[0])
            shuffle(perm)
            lperm = []
            for i in range(0, len(perm), batchsize):
                gperm = []
                for j in range(batchsize):
                    if (i + j) < len(perm):
                        gperm.append(perm[i + j])
                lperm.append(gperm)

            # for i in range(limit):
            #     yield X_train[i*batchsize:(i+1)*batchsize], y_train[i*batchsize:(i+1)*batchsize]

            for i in range(limit):
                yield X_train[lperm[i]], y_train[lperm[i]]


def dayGenerator(dataset_path, day, z_factor, nclasses, batchsize, reb=False, imgord='tf'):
    """
    Load the data for a day and returns a random permutation for
    generating the random batches

    :param day:
    :param z_factor:
    :param nclasses:
    :param batchsize:
    :return:
    """
    data, labels = load_days(dataset_path, [day], z_factor, reb=reb)
    x_train = data
    y_trainO = labels
    y_train = np_utils.to_categorical(y_trainO, nclasses)
    perm = [i for i in range(x_train.shape[0])]
    shuffle(perm)
    lperm = []
    for i in range(0, len(perm), batchsize):
        gperm = []
        for j in range(batchsize):
            if (i + j) < len(perm):
                gperm.append(perm[i + j])
        lperm.append(gperm)
    return x_train, y_train, lperm

if __name__ == '__main__':
    '''
    dataset_path = './data/Datasets/'
    ldays = list_days_generator(2016, 11, 12, 12)
    gen = simpleDataGenerator(dataset_path,ldays, 0.35, 5, 100, 5)
    for d in gen:
        print(d[0])
    '''