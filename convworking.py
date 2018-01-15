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
import ssim
import msssim
import datetime
import io
import utils
import gc

np.set_printoptions(threshold=np.nan)

class Network:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, per_image_standardization=False, batch_norm=True, skip_connections=True):
        # Define network - ENCODER (decoder will be symmetric).

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
            layers.append(MaxPool2d(kernel_size=2, name='max_3', skip_connection=skip_connections))
            
            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_4_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_4_2'))
            
            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_5_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_5_2'))

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
        
        # ENCODER
        for layer in layers:
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())

        layers.reverse()
        Conv2d.reverse_global_variables()

        # DECODER
        layers_len = len(layers)
        for i, layer in enumerate(layers):
            if i == (layers_len-1):
                self.segmentation_result = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name], last_layer=True)
            else:
                net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])

        '''# Reconstruct part
        #Input
        #net2 = tf.concat([self.segmentation_result, self.inputs], axis=3)
        net2 = self.segmentation_result
        #Layers
        slayers = []
        slayers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv2_1_1'))
        slayers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv2_1_2'))
        slayers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=1, name='conv2_2_1'))
        
        prev = None
        for l in slayers:
            old = net2
            self.layers[l.name] = net2 = l.create_layer(net2, prev_layer=prev)
            if prev == None:
                prev = old
            else:
                prev = tf.concat([prev, old], axis=3)
        self.final_result = net2'''
        self.final_result = self.segmentation_result

        # MSE loss
        # Expression Removal with MSE loss function
        output = self.segmentation_result
        inputv = self.targets
        '''tf.reshape(output, [128,128])
        tf.reshape(inputv, [128,128])
        mask1 = tf.constant(3.0, shape=[128,64])
        mask2 = tf.constant(1.0, shape=[128,64])
        mask = tf.concat([mask1, mask2], 1)'''
        mean = tf.reduce_mean(tf.square(output - inputv))
        self.cost1 = mean;
        # Reconstruct with MS_SSIM loss function
        #self.cost2 = 1 - ssim.tf_ms_ssim(self.final_result, self.targets)
        #self.cost2 = tf.reduce_mean(tf.square(self.final_result - self.targets))
        #self.cost = 50*self.cost1 + self.cost2
        self.train_op = tf.train.AdamOptimizer(learning_rate=tf.train.polynomial_decay(0.01, 1, 10000, 0.0001)).minimize(self.cost1)
        #self.train_op2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost2)
        with tf.name_scope('accuracy'):
            # argmax_probs = tf.round(self.segmentation_result)  # 0x1
            # correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            correct_pred = tf.py_func(msssim.MultiScaleSSIM, [self.final_result, self.targets], tf.float32)
            #self.accuracy = tf.reduce_mean(correct_pred)
            self.accuracy = correct_pred
            self.mse = tf.reduce_mean(tf.square(self.final_result - self.targets))

            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()


class Dataset:
    def __init__(self, batch_size, folder='data128x128', include_hair=False):
        self.batch_size = batch_size
        self.include_hair = include_hair
        
        train_files = os.listdir(os.path.join(folder, 'inputs/train'))
        validation_files = os.listdir(os.path.join(folder, 'inputs/valid'))
        test_files = os.listdir(os.path.join(folder, 'inputs/test'))

        self.train_inputs, self.train_paths, self.train_targets = self.file_paths_to_images(folder, train_files)
        self.test_inputs, self.test_paths, self.test_targets = self.file_paths_to_images(folder, test_files, mode="test")

        self.pointer = 0

    def file_paths_to_images(self, folder, files_list, mode="train"):
        inputs = []
        targets = []
        in_path = []
        test_path = []

        for file in files_list:
            input_image = os.path.join(folder, 'inputs/{}'.format(mode), file)
            output_image = os.path.join(folder, 'targets/{}'.format(mode), file)
            in_path.append(os.path.join('inputs/{}'.format(mode), file))

            test_image = cv2.imread(input_image, 0)
            test_image = cv2.resize(test_image, (128, 128))
            inputs.append(test_image)

            target_image = cv2.imread(output_image, 0)
            target_image = cv2.resize(target_image, (128, 128))
            targets.append(target_image)

        return inputs, in_path, targets

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.7, .15, .15)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )

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

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)


def draw_results(test_inputs, test_targets, test_segmentation, test_final, test_accuracy, network, batch_num):
    n_examples_to_plot = 12
    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))
    fig.suptitle("Accuracy: {}, {}".format(test_accuracy, network.description), fontsize=20)
    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i], cmap='gray')
        axs[1][example_i].imshow(test_targets[example_i].astype(np.float32), cmap='gray')
        axs[2][example_i].imshow(np.reshape(test_segmentation[example_i], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),cmap='gray')
        final = np.reshape(test_final[example_i], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH])
        blur = cv2.bilateralFilter(final, 9, 75, 75)
        blurred = np.array([0 if x < 0.1 else x for x in blur.flatten()])
        axs[3][example_i].imshow(np.reshape(blurred, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),cmap='gray')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = 'image_plots/'
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    plt.close('all')
    return buf


def train():
    BATCH_SIZE = 256

    network = Network()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # create directory for saving models
    os.makedirs(os.path.join('save', network.description, timestamp))

    dataset = Dataset(folder='data{}_{}'.format(network.IMAGE_HEIGHT, network.IMAGE_WIDTH), include_hair=False,
                      batch_size=BATCH_SIZE)

    inputs, targets = dataset.next_batch()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('{}/{}-{}'.format('logs', network.description, timestamp),
                                               graph=tf.get_default_graph())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        test_accuracies = []
        test_mse = []
        # Fit all training data
        n_epochs = 3000
        global_start = time.time()
        for epoch_i in range(n_epochs):
            dataset.reset_batch_pointer()

            for batch_i in range(dataset.num_batches_in_epoch()):
                batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1

                start = time.time()
                batch_inputs, batch_targets = dataset.next_batch()
                batch_inputs = np.reshape(batch_inputs,
                                          (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                batch_targets = np.reshape(batch_targets,
                                           (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))

                batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

                cost1, stage1, _ = sess.run([network.cost1, network.segmentation_result, network.train_op],
                                   feed_dict={network.inputs: batch_inputs, network.targets: batch_targets,
                                              network.is_training: True})
                end = time.time()
                print('{}/{}, epoch: {}, mse: {}, batch time: {}'.format(batch_num,
                                                                          n_epochs * dataset.num_batches_in_epoch(),
                                                                          epoch_i, cost1, round(end - start,5)))

                if batch_num % 100 == 0 or batch_num == n_epochs * dataset.num_batches_in_epoch():
                    test_inputs, test_targets = dataset.test_set
                    test_inputs, test_targets = test_inputs[:100], test_targets[:100]

                    test_inputs = np.reshape(test_inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                    test_targets = np.reshape(test_targets, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                    test_inputs = np.multiply(test_inputs, 1.0 / 255)

                    summary, test_accuracy, mse = sess.run([network.summaries, network.accuracy, network.mse],
                                                      feed_dict={network.inputs: test_inputs,
                                                                 network.targets: test_targets,
                                                                 network.is_training: False})

                    summary_writer.add_summary(summary, batch_num)

                    print('Step {}, test accuracy: {}'.format(batch_num, test_accuracy))
                    test_accuracies.append((test_accuracy, batch_num))
                    test_mse.append((mse, batch_num))
                    print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                    print("MSE in time: ", [test_mse[x][0] for x in range(len(test_mse))])
                    max_acc = max(test_accuracies)
                    min_mse = min(test_mse)
                    print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                    print("Total time: {}".format(time.time() - global_start))
                    
                    if mse <= min_mse[0] and mse < 5300:
                        train_dataset, paths, tam = dataset.all_train_batches()
                        for i in range(tam+1):
                            batch = train_dataset[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
                            p = paths[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
                            batch = np.multiply(batch, 1.0/255)
                            image = sess.run([network.final_result], feed_dict={network.inputs: np.reshape(batch ,[-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1])})
                            image = np.array(image[0])
                            for j in range(image.shape[0]):
                                save_image = np.resize(image[j], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH])
                                fig = plt.figure(frameon=False, dpi=100)
                                fig.set_size_inches(network.IMAGE_HEIGHT/100, network.IMAGE_WIDTH/100)
                                ax = plt.Axes(fig, [0., 0., 1., 1.])
                                ax.set_axis_off()
                                fig.add_axes(ax)
                                ax.imshow(save_image, cmap="gray")
                                fig.savefig("result/{}".format(p[j]))
                                print("salvou {}".format(p[j]))
                                plt.close(fig)
                                del fig
                                gc.collect()
                        print("All test_set exported")
                                


                    # Plot example reconstructions
                    n_examples = 12
                    test_inputs, test_targets = dataset.test_inputs[:n_examples], dataset.test_targets[:n_examples]
                    test_inputs = np.multiply(test_inputs, 1.0 / 255)

                    test_segmentation, test_final = sess.run([network.segmentation_result, network.final_result], feed_dict={
                        network.inputs: np.reshape(test_inputs,[n_examples, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1])})

                    # Prepare the plot
                    test_plot_buf = draw_results(test_inputs, test_targets, test_segmentation, test_final, test_accuracy, network,
                                                 batch_num)

                    # Convert PNG buffer to TF image
                    image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)

                    # Add the batch dimension
                    image = tf.expand_dims(image, 0)

                    # Add image summary
                    image_summary_op = tf.summary.image("plot", image)

                    image_summary = sess.run(image_summary_op)
                    summary_writer.add_summary(image_summary)

                    if test_accuracy > max_acc[0]:
                        checkpoint_path = os.path.join('save', network.description, timestamp, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=batch_num)


if __name__ == '__main__':
    train()
