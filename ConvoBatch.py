'''
.. module:: ConvoBatch

ConvoBatch
*************

  Trains a model according to a configuration file (--batch) or the harcoded config object
  Model is trained using the train_on_batch method from Keras model, so only a day is loaded in memory at a time

:Description: ConvoBatch

:Authors: shreyas

:Version: 

:Created on: 23/12/2016 15:05 

'''


from keras import backend as K
from keras.optimizers import SGD, Adagrad, Adadelta, Adam
from SimpleModels import simple_model
from ConvoTrain import transweights, detransweights, train_model_batch
from DataGenerators import list_days_generator, dayGenerator
from ConvoTrain import load_dataset
import datetime
__author__ = 'shreyas'

if __name__ == '__main__':

    ldaysTr = list_days_generator(2016, 11, 7, 13)
    ldaysTs = list_days_generator(2016, 12, 1, 2)
    z_factor = 0.35
    #classweight = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    classweight = {0: 2.0, 1: 1.0, 2: 4.0, 3: 8.0, 4: 16.0}
    config = {
        'datasetpath': './data/Datasets/',
        'savepath': './data/Models/',
        'traindata': ldaysTr,
        'testdata': ldaysTs,
        'rebalanced': False,
        'zfactor': z_factor,
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
                 {'lrate': 0.0001,
                  'momentum': 0.9,
                  }},
        "train":
            {"batchsize": 256,
             "epochs": 1,
             "classweight": transweights(classweight)},

        'imgord': 'th'
    }
    K.set_image_dim_ordering(config['imgord'])
    config['datasetpath'] += config['imgord']
    config['datasetpath'] += '/'
    train, test, test_labels, num_classes = load_dataset(config, only_test=False)
    config['input_shape'] = test[0][0].shape
    config['num_classes'] = num_classes
    classweight = detransweights(config['train']['classweight'])
    model = simple_model(config)
    begin = datetime.datetime.now()
    #train_model_batch(model, config, test, test_labels)
    if config['optimizer']['method'] == 'adagrad':
        optimizer = Adagrad()
    elif config['optimizer']['method'] == 'adadelta':
        optimizer = Adadelta()
    elif config['optimizer']['method'] == 'adam':
        optimizer = Adam()
    else:  # default SGD
        params = config['optimizer']['params']
        optimizer = SGD(lr=params['lrate'], momentum=params['momentum'], decay=params['lrate'] / params['momentum'],
                        nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(train[0],train[1], batch_size=config['train']['batchsize'], nb_epoch=config['train']['epochs'], class_weight=classweight,verbose=1)
    model.save(config['savepath'] + '/model' + '.h5')
    scores = model.evaluate(test[0], test[1], verbose=0)
    y_pred = model.predict_classes(test[0], verbose=0)
    print 'begin %s' % begin
    print 'end %s' % datetime.datetime.now()
    print classweight
    print scores
    print len(y_pred)
    ones = 0
    zeroes = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            ones += 1
        elif y_pred[i] == 0:
            zeroes += 1
        else:
            print y_pred[i]
    print "no of ones %d" % ones
    print "no of zeroes %d" % zeroes
    print "----------------------------------------------------------------------------------------------------------------"