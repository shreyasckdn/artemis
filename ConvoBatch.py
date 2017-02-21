'''
.. module:: ConvoBatch

ConvoBatch
*************

  Trains a model according to a configuration file (--batch) or the harcoded config object
  Model is trained using the train_on_batch method from Keras model, so only a day is loaded in memory at a time

:Description: ConvoBatch

:Authors: bejar

:Version: 

:Created on: 23/12/2016 15:05 

'''


from keras import backend as K
from SimpleModels import simple_model
from ConvoTrain import transweights, train_model_batch
from DataGenerators import list_days_generator
from ConvoTrain import load_dataset

import json
import argparse

__author__ = 'bejar'

if __name__ == '__main__':

    ldaysTr = list_days_generator(2016, 11, 7, 13)
    ldaysTs = list_days_generator(2016, 12, 1, 2)
    z_factor = 0.25
    camera = None  # 'Ronda' #Cameras[0]

    smodel = 3
    classweight = {0: 2.0, 1: 1.0, 2: 4.0, 3: 8.0, 4: 16.0}

    config = {
        'datapath': './data/Datasets/',
        'savepath': './data/Models/',
        'traindata': ldaysTr,
        'testdata': ldaysTs,
        'rebalanced': False,
        'zfactor': 0.35,
        'model': 4,
        'convolayers':
            {'sizes': [128, 64, 32],
             'convofields': [3, 3],
             'dpconvo': 0.2,
             'pool': ['max', 2, 2]},
        'fulllayers':
            {'sizes': [64, 32],
             'reg': ['l1', 0.2]},
        'optimizer':
            {'method': 'sdg',
             'params':
                 {'lrate': 0.005,
                  'momentum': 0.9,
                  }},
        "train":
            {"batchsize": 256,
             "epochs": 50,
             "classweight": transweights(classweight)},

        'imgord': 'tf'
    }

        # config['optimizer']['params']['decay'] = config['lrate'] / config['epochs']

    K.set_image_dim_ordering(config['imgord'])

    _, test, test_labels, num_classes = load_dataset(config, only_test=True, imgord=config['imgord'])
    config['input_shape'] = test[0][0].shape
    config['num_classes'] = num_classes

    model = simple_model(config)
    print 'begin'
    train_model_batch(model, config, test, test_labels)
    print 'end'