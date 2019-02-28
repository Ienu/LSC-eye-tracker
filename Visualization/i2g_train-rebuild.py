# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
    Author: Xu Yucheng
    Date: 02/19/2019
    Env: Keras 2.2.4

    function:
    rebuild by keras Model API
'''
import os
import numpy as np 
import scipy.io as sio
import h5py
import keras
import math
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import Callback
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPool2D,
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

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

class feature_map_vis(Callback):
    """TensorBoard basic visualizations.
    log_dir: the path of the directory where to save the log
        files to be parsed by TensorBoard.
    write_graph: whether to visualize the graph in TensorBoard.
        The log file can become quite large when
        write_graph is set to True.
    batch_size: size of batch of inputs to feed to the network
        for histograms computation.
    input_images: input data of the model, because we will use it to build feed dict to
        feed the summary sess.
    write_features: whether to write feature maps to visualize as
        image in TensorBoard.
    update_features_freq: update frequency of feature maps, the unit is batch, means
        update feature maps per update_features_freq batches
    update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
        the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `10000`,
        the callback will write the metrics and losses to TensorBoard every
        10000 samples. Note that writing too frequently to TensorBoard
        can slow down your training.
    """

    def __init__(self, log_dir='./logs',
                 batch_size=64,
                 update_features_freq=1,
                 input_images=None,
                 write_graph=True,
                 write_features=False,
                 update_freq='epoch'):
        super(feature_map_vis, self).__init__()
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError("You need to install tensorflow to use tensorboard")

        if K.backend() != 'tensorflow':
            if write_graph:
                warnings.warn('You are not using the TensorFlow backend. '
                              'write_graph was set to False')
                write_graph = False
            if write_features:
                warnings.warn('You are not using the TensorFlow backend. '
                              'write_features was set to False')
                write_features = False

        self.input_images = input_images[0]
        self.log_dir = log_dir
        self.merged = None
        self.image_summary = []
        self.lr_summary = None
        self.write_graph = write_graph
        self.write_features = write_features
        self.batch_size = batch_size
        self.update_features_freq = update_features_freq
        if update_freq == 'batch':
            # It is the same as writing as frequently as possible.
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self.samples_seen = 0
        self.samples_seen_at_last_write = 0

    def _concat_features(self, conv_output):
        """Concat operation for feature map
        Param: conv_output
        return: concated feature map
        """
        all_concat = None
        
        num_or_size_split = conv_output.get_shape().as_list[-1]
        each_filter = tf.split(conv_output, num_or_size_split=num_or_size_split)

        if num_or_size_split < 4:
            # If the number of channel of feature map is less than4
            # Then regard it as the input images
            # Only need to concatenate the images
            concat_size = num_or_size_split
            all_concat = each_filter[0]
            for i in range (concat_size - 1):
                all_concat = tf.concat([all_concat, each_filter[i+1]], 1)

        else:
            concat_size = int (math.sqrt(num_or_size_split)/1)
            for i in range (concat_size):
                row_concat = each_filter[i*concat_size]
                for j in range (concat_size-1):
                    row_concat = tf.concat([row_concat, each_filter[i*concat_size+1]], 1)
                    if i == 0:
                        all_concat == row_concat
                    else:
                        all_concat = tf.concat([all_concat, row_concat], 2)
        return all_concat

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            # Get current tensorflow session
            self.sess = K.get_session()
        if self.merged is None:
            # Feature map visualization
            # Ergodic our layers
            for layer in self.model.layers:
                # Get the name and output of current layer
                feature_map = layer.output
                feature_map_name = layer.name.replace(':', '_')

                if sself.write_features and len(K.int_shape(feature_map)) ==4:
                    # Flat and concat feature map
                    flat_concat_feature_map = self._concat_features(feature_map)
                    # Make sure the channel of feature map is 1]
                    shape = K.int_shape(flat_concat_feature_map)
                    assert len(shape) == 4 and shape[-1]==1
                    # Write feature map to tensorboard
                    self.image_summary.append(tf.summary.image(feature_map_name, flat_concat_feature_map))
            
            # Visualization the change of learning rate
            self.lr_summary = tf.summary.scalar("learning_rate", self.model.optimizer.lr)
        self.merged = tf.summary.merge_all()
    
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summart.FileWriter(self.log_dir)

    def on_train_end(self):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or ()
        # On epoch end we use valdation data to validate our net
        # Here is the function to record output of
        if self.validation_data:
            val_data = self.validation_data
            tensors = (self.model.inputs + 
                       self.model.targets +
                       self.model.sample_weights)
            
            # Use 'self.model.learning_phase'
            # To judge this net is working on train mode or test mode
            if self.model.use_learning_phase:
                tensors += [K.learning_phase()]
            
            assert len(val_data) == len(tensors)
            val_size = val_data[0].shape[0]
            i = 0
            while i < val_size:
                step = min(self.batch_size, val_size-1)
                if self.model.use_learning_phase:
                    # do not slice the learning phase
                    batch_val = [x[i:i + step] for x in val_data[:-1]]
                    batch_val.append(val_data[-1])
                else:
                    batch_val = [x[i:i + step] for x in val_data]
                assert len(batch_val) == len(tensors)
                feed_dict = dict(zip(tensors, batch_val))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)
                i += self.batch_size

        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        self._write_logs(logs, index)

    def on_batch_end(self, batch, logs=None):
        if self.update_freq != 'epoch':
            self.samples_seen += logs['size']
            samples_seen_since = self.samples_seen - self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:
                self._write_logs(logs, self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen

        # 每update_features_freq个batch刷新特征图
        if batch % self.update_features_freq == 0:
            # 计算summary_image
            feed_dict = dict(zip(self.model.inputs, self.input_images[np.newaxis, ...]))
            for i in range(len(self.image_summary)):
                summary = self.sess.run(self.image_summary[i], feed_dict)
                self.writer.add_summary(summary, self.samples_seen)

        # 每个batch显示学习率
        summary = self.sess.run(self.lr_summary, {self.model.optimizer.lr: K.eval(self.model.optimizer.lr)})
        self.writer.add_summary(summary, self.samples_seen)
