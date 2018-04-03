import math
import os
import time
from math import ceil

import cv2
import sys
import matplotlib

from scipy import signal
from scipy import ndimage

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from PIL import Image

from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import datetime
import io
import utils
import gc
import tensorflow.contrib.slim as slim

np.set_printoptions(threshold=np.nan)

class Network:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, skip_connections=False):
        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_1_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=skip_connections))

            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_2_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_2_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=skip_connections))

            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_3_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_2'))            
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_3'))
            
        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        net = self.inputs
        
        # ENCODER
        for layer in layers:
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())

        layers.reverse()
        Conv2d.reverse_global_variables()   
        
        #midfield
        old_shape = net.get_shape()
        o_s = old_shape.as_list()
        feature_len = o_s[1]*o_s[2]*o_s[3]
        for i in range(3):
            net = tf.reshape(net, [-1, feature_len])
            net = slim.fully_connected(net, feature_len, scope="fc_{}".format(i+1))
            net = tf.reshape(net, [-1,old_shape[1],old_shape[2],old_shape[3]])
            toAdd = net
            net = slim.repeat(net, 2, slim.conv2d, 1, [3,3], scope="block{}".format(i+1))
            net = tf.add(toAdd, net)

        # DECODER
        layers_len = len(layers)
        for i, layer in enumerate(layers):
            if i == (layers_len-1):
                self.segmentation_result = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name], last_layer=True)
            else:
                net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])

        self.final_result = self.segmentation_result

class Dataset:
    def __init__(self, batch_size, folder='data128x128'):
        self.batch_size = batch_size
        
        train_files = os.listdir(folder)

        self.train_inputs, self.train_paths = self.file_paths_to_images(folder, train_files)

        self.pointer = 0

    def file_paths_to_images(self, folder, files_list):
        inputs = []
        in_path = []

        for file in files_list:
            input_image = os.path.join(folder, file)
            in_path.append(file)

            test_image = cv2.imread(input_image, 0)
            test_image = cv2.resize(test_image, (128, 128))
            inputs.append(test_image)

        return inputs, in_path

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_paths = [self.train_paths[i] for i in permutation]

        self.pointer = 0
        
    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def next_batch(self):
        inputs = []
        paths = []
        
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            paths.append(self.train_paths[self.pointer + i])

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), paths

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)

def saveImage(image, height, width, path):
    fig = plt.figure(frameon=False, dpi=100)
    fig.set_size_inches(height/100, width/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, cmap="gray")
    fig.savefig(path)
    plt.close(fig)
    del fig
    gc.collect()

def load():
    network = Network()
    saver = tf.train.Saver()
    
    dataset = Dataset(folder='data128_128/targets/test/', batch_size=1)
    
    config = tf.ConfigProto(
            device_count = {'GPU' : 0}
    )
    
    with tf.Session(config=config) as sess:
        saver.restore(sess, "save/checkpoint.data-0")
        print("Model Restored")
        
        dataset.reset_batch_pointer()
        count = 0
        
        for batch_i in range(dataset.num_batches_in_epoch()):
            batch_inputs, paths = dataset.next_batch()
            batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
            batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

            start = time.time()
            image = sess.run([network.segmentation_result], feed_dict={network.inputs: batch_inputs,network.is_training: False})
            end = time.time()
            total_time = round(end-start,5)
            image = np.array(image[0])
            for j in range(image.shape[0]):
                save_image = np.resize(image[j], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH])
                path = "result/gt_rede/{}/{}".format("new",paths[j])
                saveImage(save_image, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, path)
                print("Salvando {} em {}s ({} de {})".format(paths[j], total_time, count, len(dataset.train_inputs)))
                count += 1
        print("All images exported")
        

if __name__ == '__main__':
    load()
