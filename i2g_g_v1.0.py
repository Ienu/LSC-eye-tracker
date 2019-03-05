import numpy as np
import scipy.io as sio
import h5py
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import os

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
        plt.savefig('%(index)d.png'%{'index': index})

history = LossHistory()

# this is for k-folder cross validation
seed = 7
np.random.seed(seed)

# init
batch_size = 32
num_classes = 4
epochs = 20

# data load and preprocess
print(sys.argv[1])
f = h5py.File(sys.argv[1])

x = np.array(f['faceData'])
y = np.array(f['eyeTrackData'])

x = np.transpose(x, (3, 2, 1, 0))
y = np.transpose(y, (1, 0))

print('x shape: ', x.shape)
print('y shape: ', y.shape)


x = x.astype('float32')
x /= 255
y = y.astype('float32')

# k-folder cross validation
kf = KFold(n_splits=5)
cvscores = []
model_index = 1
for train, test in kf.split(x, y):
    # model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=x.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_classes))

    model.summary()

    # train
    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    model.fit(x[train], y[train],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          callbacks=[history])

    history.loss_plot(model_index, 'epoch')

    model.save('model%(index)d.h5'%{'index':model_index})
    model_index = model_index + 1
    scores = model.evaluate(x[test], y[test], verbose=0)
    print('Test loss:', scores[0], scores[1])
    cvscores.append(scores[0])

print('Mean loss:', np.mean(cvscores))
for i in range(0, 5):
    print(cvscores[i])
