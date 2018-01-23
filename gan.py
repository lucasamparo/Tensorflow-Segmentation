import os
import sys
import json
import fnmatch
import tarfile
from PIL import Image
from glob import glob
from tqdm import tqdm
from six.moves import urllib

import numpy as np

class Dataset:
    def __init__(self, batch_size, train_path, test_path):
        self.batch_size = batch_size
        self.include_hair = include_hair
        
        train_files = os.listdir(train_path)
        test_files = os.listdir(test_path)

        self.train_inputs, self.train_paths, self.train_targets = self.file_paths_to_images(train_path, train_files)
        self.test_inputs, self.test_paths, self.test_targets = self.file_paths_to_images(test_path, test_files)

        self.pointer = 0

    def file_paths_to_images(self, folder, files_list):
        inputs = []
        targets = []
        in_path = []
        test_path = []

        for file in files_list:
            input_image = os.path.join(folder, file)
            output_image = os.path.join(folder, file)
            in_path.append(os.path.join(file))

            test_image = cv2.imread(input_image, 0)
            test_image = cv2.resize(test_image, (128, 128))
            inputs.append(test_image)

            target_image = cv2.imread(output_image, 0)
            target_image = cv2.resize(target_image, (128, 128))
            targets.append(target_image)

        return inputs, in_path, targets

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_paths = [self.train_inputs[i] for i in permutation]
        self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        targets = []
        
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)
        
    def all_train_batches(self):
        return np.array(self.test_inputs, dtype=np.uint8), self.test_paths, len(self.test_inputs)//self.batch_size

class Refinador:
    MAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, per_image_standardization=False, batch_norm=True, skip_connections=True):
        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_1'))
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2'))
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_3'))
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_4'))
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_5'))
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=1, name='conv_2_1'))
            
        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        if per_image_standardization:
            list_of_images_norm = tf.map_fn(tf.image.per_image_standardization, self.inputs)
            net = tf.stack(list_of_images_norm)
        else:
            net = self.inputs
        
        for layer in layers:
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())
            
        self.final_result = tanh(net, name="output")

        # MSE loss
        output = self.segmentation_result
        inputv = self.targets
        mean = tf.reduce_mean(tf.square(output - inputv))
        self.cost1 = mean;
        self.train_op = tf.train.AdamOptimizer(learning_rate=tf.train.polynomial_decay(0.01, 1, 10000, 0.0001)).minimize(self.cost1)
        with tf.name_scope('accuracy'):
            correct_pred = tf.py_func(msssim.MultiScaleSSIM, [self.final_result, self.targets], tf.float32)
            self.accuracy = correct_pred
            self.mse = tf.reduce_mean(tf.square(self.final_result - self.targets))

            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()
