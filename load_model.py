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
from recog_model import inception_resnet_v1 as model
import load_recog as recog

np.set_printoptions(threshold=np.nan)

class Network:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, skip_connections=False, has_translator=True):
        with tf.variable_scope('encoder-decoder'):
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

            self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS], name='inputs')
            self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
            self.is_training = tf.placeholder_with_default(False, [], name='is_training')
            self.description = ""

            self.layers = {}

            net = self.inputs

            # ENCODER
            for layer in layers:
                    self.layers[layer.name] = net = layer.create_layer(net, is_training=False)

            layers.reverse()
            Conv2d.reverse_global_variables()    

            #midfield
            if(has_translator == True):
                old_shape = net.get_shape()
                o_s = old_shape.as_list()
                feature_len = o_s[1]*o_s[2]*o_s[3]
                net = tf.reshape(net, [-1, feature_len])
                for i in range(3):
                    net = slim.fully_connected(net, feature_len, scope="fc_{}".format(i+1))
                    self.fc_vars = tf.contrib.framework.get_variables("fc_{}".format(i+1))
                net = tf.reshape(net, [-1, o_s[1], o_s[2], o_s[3]])

            # DECODER
            layers_len = len(layers)
            for i, layer in enumerate(layers):
                    if i == (layers_len-1):
                            self.segmentation_result = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name], last_layer=True, is_training=False)
                    else:
                            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name], is_training=False)

        self.final_result = self.segmentation_result
        self.variables = tf.contrib.framework.get_variables(scope='encoder-decoder')

class Dataset:
    def __init__(self, batch_size, folder='data128x128'):
        self.batch_size = batch_size
        
        all_paths = self.open_folder(folder)
        self.train_inputs, self.train_paths = self.file_paths_to_images(all_paths)

        self.pointer = 0

    def open_folder(self, folder):
        train_files = os.listdir(folder)
        all_paths = []
        for file in train_files:
            file_path = os.path.join(folder,file)
            if(os.path.isfile(file_path)):
                all_paths.append(file_path)
            else:
                all_paths.extend(self.open_folder(file_path))
        return all_paths


    def file_paths_to_images(self, files_list):
        inputs = []
        in_path = []

        for file in files_list:
            in_path.append(file)
            test_image = cv2.imread(file, 0)
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

def create_dirs(model_path):
    if(not os.path.exists(model_path+"/inputs")):
        os.makedirs(os.path.join(model_path, "inputs/valid"))
        os.makedirs(os.path.join(model_path, "inputs/test"))
        os.makedirs(os.path.join(model_path, "inputs/train"))

    if(not os.path.exists(model_path+"/targets")):
        os.makedirs(os.path.join(model_path, "targets/valid"))
        os.makedirs(os.path.join(model_path, "targets/test"))
        os.makedirs(os.path.join(model_path, "targets/train"))

    print("All directories created")

def load(model_path, has_translator):
    network = Network(has_translator=has_translator)
    saver = tf.train.Saver()
    
    dataset = Dataset(folder='data128_128/', batch_size=128)
    
    config = tf.ConfigProto(
            device_count = {'GPU' : 0}
    )
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "saved_models/{}/checkpoint.data".format(model_path))
        #saver.restore(sess, "save/checkpoint.data")
        print("General Model Restored")
        create_dirs(os.path.join("result/export", model_path))
        
        dataset.reset_batch_pointer()
        count = 0

        '''variables = sess.run(network.variables)

        with open("variables_out.txt", "w") as f:
            for v in variables:
                print("Variable: ", v, file=f)
        print("All variables saved")
        quit()'''

        
        for batch_i in range(dataset.num_batches_in_epoch()):
            batch_inputs, paths = dataset.next_batch()
            batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
            batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

            start = time.time()
            image = sess.run([network.segmentation_result], feed_dict={network.inputs: batch_inputs,network.is_training: True})
            end = time.time()
            total_time = round(end-start,5)
            image = np.array(image[0])
            for j in range(image.shape[0]):
                save_image = np.squeeze(image[j])
                split_path = paths[j].split("/")
                path = "result/export/{}/{}/{}".format(model_path,os.path.join(split_path[1],split_path[2]), split_path[3])
                saveImage(save_image, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, path)
                print("Salvando {} em {}s ({} de {})".format(split_path[1]+"/"+split_path[2]+"/"+split_path[3], total_time, count, len(dataset.train_inputs)))
                count += 1
        print("All images exported")
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
        type=str,
        required=True,
        help='Caminho do Modelo')

    parser.add_argument('--has_translator',
        type=str,
        default=True,
        help='Se existe tradutor')

    FLAGS = parser.parse_args()
    
    load(FLAGS.model_path, FLAGS.has_translator)
