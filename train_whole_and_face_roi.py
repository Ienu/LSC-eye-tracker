# -*- coding: utf-8 -*-
"""
    Author: Wenyu
    Date: 2/28/2019
    Version: 4.1
    Env: Python 3.7, Keras 2.2.4
    
    Function:
    v1.0[insfan][4/29/2019]:whole image and face image are trained at the same time.
"""
import numpy as np
import scipy.io as sio
import h5py
import sys
import keras
import keras.backend as kb
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import os
import time

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

def create_model_common(x):
    """
        model regress direct to the gaze point
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

    model.add(Dense(2))

    return model


def create_model_merge(scale):
    """
        merge two models
    """
    # whole img input conv op
    w_input = Input(shape=(224, 224, 3), dtype='float32', name='whole_input')
    x_wh = Conv2D(16, (3, 3), padding='same')(w_input)
    x_wh = LeakyReLU()(x_wh)
    x_wh = MaxPooling2D(pool_size=(2, 2))(x_wh)
    x_wh = BatchNormalization()(x_wh)

    x_wh = Conv2D(32, (3, 3), padding='same')(x_wh)
    x_wh = LeakyReLU()(x_wh)
    x_wh = MaxPooling2D(pool_size=(2, 2))(x_wh)
    x_wh = BatchNormalization()(x_wh)

    x_wh = Conv2D(64, (3, 3), padding='same')(x_wh)
    x_wh = LeakyReLU()(x_wh)
    x_wh = MaxPooling2D(pool_size=(2, 2))(x_wh)
    x_wh = BatchNormalization()(x_wh)

    # face roi input conv op
    f_input = Input(shape=(224, 224, 3), dtype='float32', name='face_input')
    x_fe = Conv2D(16, (3, 3), padding='same')(f_input)
    x_fe = LeakyReLU()(x_fe)
    x_fe = MaxPooling2D(pool_size=(2, 2))(x_fe)
    x_fe = BatchNormalization()(x_fe)

    x_fe = Conv2D(32, (3, 3), padding='same')(x_fe)
    x_fe = LeakyReLU()(x_fe)
    x_fe = MaxPooling2D(pool_size=(2, 2))(x_fe)
    x_fe = BatchNormalization()(x_fe)

    x_fe = Conv2D(64, (3, 3), padding='same')(x_fe)
    x_fe = LeakyReLU()(x_fe)
    x_fe = MaxPooling2D(pool_size=(2, 2))(x_fe)
    x_fe = BatchNormalization()(x_fe)

    # merge whole img conv and face img conv
    x = keras.layers.concatenate([x_wh, x_fe])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(scale * scale * 3)(x)
    x = Activation('tanh')(x)
    x = Reshape((scale, scale, 3))(x)

    model = Model(inputs=[w_input, f_input], outputs=[x])
    return model


        
def main():
    """
        main function
        argv[1]: data file name (.npz)
        argv[2]: model name for saving
        [argv[3]]: model name for loading
    """
    assert len(sys.argv) >= 2, 'Not enough arguments'
    
    # Warning
    print('################################################################')
    print('# WARNING: The Code Should Be Tested On A Small Dataset First! #')
    print('################################################################')

    name = sys.argv[2]

    # history = LossHistory()

    # init
    batch_size = 64
    scale = 7
    epochs = 50

    # data load and preprocess
    f = np.load(sys.argv[1])

    x_w = np.array(f['faceData'])
    x_f = np.array(f['faceRoiData'])
    y = np.array(f['eyeTrackData'])

    print('x shape: ', x_w.shape)
    print('y shape: ', y.shape)

    x_w = x_w.astype('float32')
    x_w /= 255
    x_f = x_f.astype('float32')
    x_f /= 255
    y = y.astype('float32')

    amount = x_w.shape[0]
    print('Sample amount: ', amount)

    # model
    # TODO: a pretrain needs to be considered on ImageNet or some datasets else
    # or init the params with existing ImageNet's weights

    if len(sys.argv) == 4:
        model = load_model(sys.argv[3])
    else:    
        model = create_model_merge(scale)

    model.summary()

    # train
    opt = keras.optimizers.Adam(lr=0.001)
    # loss function form should be considered
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    # consider EarlyStopping [hard to use]
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit([x_w, x_f], [y],
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     shuffle=True,
                     validation_split=0.2)
                     #callbacks=[early_stopping])

    # model_index = 0
    # history.loss_plot(model_index, 'epoch')
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.sqrt(history.history['loss'])*1000, color='b', label="Training loss")
    ax[0].plot(np.sqrt(history.history['val_loss'])*1000, color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(np.sqrt(history.history['acc'])*1000, color='b', label="Training accuracy")
    ax[1].plot(np.sqrt(history.history['val_acc'])*1000, color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.savefig('loss_whole_and_face.png')
    model.save('model_%s.h5' % name)

    with open('log%s.txt',time.time()) as f:
        f.write(str(history.history()))


if __name__ == '__main__':
    main()
