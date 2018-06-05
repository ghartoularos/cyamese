from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.models import Model
from keras.layers import (Input, Flatten, Dense, Dropout, Lambda,
                          Conv2D, AveragePooling2D, MaxPooling2D,
                          Conv1D, Activation)
from keras.utils import Sequence
from time import localtime, strftime
import os
import random
import numpy as np
import subprocess

class DataGenerator(object):
    """docstring for DataGenerator"""
    def __init__(self,pkls):
        filenames  = os.listdir(pkls)
        self.path = pkls
        self.train_names = [f for f in filenames if 'tr' in f]
        self.test_names = [f for f in filenames if 'te' in f]
        self.train_steps = len(self.train_names)
        self.test_steps = len(self.test_names)

    def next_train(self):
        while 1:
            random.shuffle(self.train_names)
            for filename in self.train_names:
                data = np.load(self.path + '/' + filename)
                if ".pos." in filename:
                    label = np.ones((len(data)))
                else:
                    label = np.zeros((len(data)))
                data_t = np.transpose(data,(1,0,2,3,4))
                yield ([data_t[0],data_t[1]],label)
    def next_test(self):
        while 1:
            random.shuffle(self.test_names)
            for filename in self.test_names:
                data = np.load(self.path + '/' + filename)
                if ".pos." in filename:
                    label = np.ones((len(data)))
                else:
                    label = np.zeros((len(data)))
                data_t = np.transpose(data,(1,0,2,3,4))
                yield ([data_t[0],data_t[1]],label)

def create_base_network(input_shape, width):
    modinput = Input(shape=input_shape)
    x=Conv2D(filters=32, 
             kernel_size=(1, width),
             activation='relu')(modinput)
    x = AveragePooling2D(pool_size=(input_shape[0], 1))(x)
    x = Flatten()(x)
    x = Dense(20, activation='relu')(x)
    return Model(modinput, x)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def getmodel(input_shape, width):
    # network definition
    # raw_input(input_shape)
    base_network = create_base_network(input_shape, width)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    return model

def trainCNN(pathtopkl, model, epochs, loadbar, gpu_switch):
    rms = Adam()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    if loadbar:
        verbose = 1
    else:
        verbose = 2
    datagen = DataGenerator(pathtopkl)
    modelFit = model.fit_generator(generator=datagen.next_train(),
                                   epochs = epochs,
                                   steps_per_epoch=datagen.train_steps,
                                   validation_data=datagen.next_test(),
                                   validation_steps=datagen.test_steps,
                                   verbose=verbose)
    import matplotlib as mpl
    if gpu_switch:
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    randint = random.randint(0,1000)
    print("Result ID: MMDD-%d" % randint)
    foldername = '%s/result_%s-%d' % (pathtopkl, strftime("%m%d", localtime()), randint)
    shellcommand = ['mkdir', foldername]
    p = subprocess.Popen(shellcommand, stdout=subprocess.PIPE)
    plt.plot(modelFit.history['accuracy'])
    plt.plot(modelFit.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('%s/accuracy.png' % foldername)
    plt.close()
    plt.plot(modelFit.history['loss'])
    plt.plot(modelFit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('%s/loss.png' % foldername)
    return 
