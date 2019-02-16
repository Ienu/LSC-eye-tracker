# -*- coding: utf-8 -*-
"""
    Author: Wenyu
    Date: 2/16/2019
    Version: 4.0
    Env: Keras 2.2.4
    
    Function:
    v3.1: add cross-validation
    v3.2: structured and add some comments
    v4.0: can load saved model for continously training
"""
import numpy as np
import scipy.io as sio
import h5py
import sys
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import os

# TODO: we should consider runing the code with the original data directly
# TODO: We should consider our mAP
# TODO: Speed should be considered in the future

# TODO: Due to YOLO v2, we can extend our capacity with different faces
# Face related research should be considered

# TODO: divide this class into another file
# this class is for display loss graph
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, index, loss_type):
        iters = range(len(self.losses[loss_type]))

        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        #plt.show()
        plt.savefig('%(name)d.png'%{'name': name})


def create_model(x, scale):
    """
        model
    """
    model = Sequential()

    # according to YOLO v2, batch normalization can replace dropout layer
    # so we remove dropout in order to train faster

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=x.shape[1:]))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(LeakyReLU())
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (1, 1), padding='same'))
    model.add(LeakyReLU())
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU())
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(scale * scale * 3))
    model.add(Activation('tanh'))

    model.add(Reshape((scale, scale, 3)))
    #model.summary()

    return model

        
def main():
    """
        main function
        argv[1]: data file name (.npz)
        argv[2]: model name for saving
        argv[3]: model name for loading
    """
    # Warning
    print('################################################################')
    print('# WARNING: The Code Should Be Tested On A Small Dataset First! #')
    print('################################################################')

    name = sys.argv[2]

    history = LossHistory()

    # this is for k-folder cross validation
    seed = 7
    np.random.seed(seed)

    # init
    batch_size = 64
    #num_classes = 2
    scale = 7
    epochs = 1000
    patch = 100

    # data load and preprocess
    print(sys.argv[1])
    f = np.load(sys.argv[1])

    x = np.array(f['faceData'])
    y = np.array(f['eyeTrackData'])

    print('x shape: ', x.shape)
    print('y shape: ', y.shape)

    x = x.astype('float32')
    x /= 255
    y = y.astype('float32')

    amount = x.shape[0]
    print('Sample amount: ', amount)

    # model
    # TODO: a pretrain needs to be considered on ImageNet or some datasets else
    # or init the params with existing ImageNet's weights

    if len(sys.argv) == 4:
        model = load_model(sys.argv[3])
    else:    
        model = create_model(x, scale)

    # train
    opt = keras.optimizers.rmsprop(lr=0.001, decay=0.0001)
    # loss function form should be considered
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    # TODO: save model every 100 epochs
    for ep in range(patch, epochs, patch):
        print('Epoch %d' % ep)
        model.fit(x[:int(amount*9/10),],
                  y[:int(amount*9/10),],
                  batch_size=batch_size,
                  epochs=patch,
                  verbose=1,
                  shuffle=True)#,
                  #callbacks=[history])
        scores = model.evaluate(x[int(amount*9/10)+1:,], 
                                y[int(amount*9/10)+1:,],
                                verbose=1)
        print('Test loss:', scores[0], scores[1])
        model.save('model_%s_%d.h5' % (name, ep))

    #model_index = 0
    #history.loss_plot(model_index, 'epoch')

    model.save('model_%s_last.h5' % name)
        
    scores = model.evaluate(x[int(amount*9/10)+1:,], 
                            y[int(amount*9/10)+1:,],
                            verbose=1)
    print('Test loss:', scores[0], scores[1])


if __name__ == '__main__':
    main()
