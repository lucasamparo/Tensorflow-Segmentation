import os
import sys
import io
import json
import fnmatch
import tarfile
import math
from PIL import Image
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import sample

from conv2d import Conv2d
from max_pool_2d import MaxPool2d

import numpy as np
import tensorflow as tf
import cv2

class Dataset:
    def __init__(self, batch_size, input_path, ground_truth, test_input_path, test_ground_truth):
        self.batch_size = batch_size
        
        train_files = os.listdir(input_path)
        train_gt = os.listdir(ground_truth)
        test_files = os.listdir(test_input_path)
        test_gt = os.listdir(test_ground_truth)

        self.train_inputs, self.train_paths = self.file_paths_to_images(input_path, train_files)
        self.train_outputs, self.troutput_paths = self.file_paths_to_images(ground_truth, train_gt)
        
        self.test_inputs, self.test_paths = self.file_paths_to_images(test_input_path, test_files)
        self.test_outputs, self.tsoutput_paths = self.file_paths_to_images(test_ground_truth, test_gt)

        self.pointer = 0
        
        self.history_buf = self.train_inputs

    def file_paths_to_images(self, folder, files_list):
        inputs = []
        in_path = []

        for file in files_list:
            input_image = os.path.join(folder, file)
            in_path.append(os.path.join(file))

            test_image = cv2.imread(input_image, 0)
            test_image = cv2.resize(test_image, (128, 128))
            inputs.append(test_image)

        return inputs, in_path

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_paths = [self.train_paths[i] for i in permutation]
        permutation = np.random.permutation(len(self.train_outputs))
        self.train_outputs = [self.train_outputs[i] for i in permutation]
        self.troutput_paths = [self.troutput_paths[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        targets = []
        
        for i in range(self.batch_size):
            if i <= (self.batch_size/2):
                inputs.append(np.array(self.train_inputs[self.pointer + i]))
                targets.append(np.array(self.train_outputs[self.pointer + i]))
            else:
                inputs.append(np.array(self.history_buf[self.pointer + i]))
                targets.append(np.array(self.train_outputs[self.pointer + i]))            

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)
        
    def all_test_batches(self):
        return np.array(self.test_inputs, dtype=np.uint8), self.test_paths, len(self.test_inputs)//self.batch_size
        
    def get_hist_batch(self):
        inputs = []
        lista = sample(range(len(self.train_inputs)), self.batch_size)
        for i in lista:
            inputs.append(np.array(self.train_inputs[i]))
            
        return np.array(inputs, dtype=np.uint8), lista
        
    def set_hist_batch(self, tensor, lista):
        i = 0;
        tensor = np.array(tensor[0])
        for j in range(tensor.shape[0]):
            self.history_buf[lista[i]] = cv2.resize(tensor[j], (128,128))
            i += 1
        
        
class Discriminador:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, name="", skip_connections=True, external_input=None):
        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=3, strides=[1, 2, 2, 1], output_channels=96, name='dconv_1_1'))
            layers.append(Conv2d(kernel_size=3, strides=[1, 2, 2, 1], output_channels=64, name='dconv_1_2'))
            layers.append(MaxPool2d(kernel_size=3, name='max_1', skip_connection=skip_connections))
            
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=32, name='dconv_2_1'))
            layers.append(Conv2d(kernel_size=1, strides=[1, 1, 1, 1], output_channels=32, name='dconv_2_2'))
            layers.append(Conv2d(kernel_size=1, strides=[1, 1, 1, 1], output_channels=2, name='dconv_2_3'))
            
        if external_input == None:
            self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],name='{}_inputs'.format(name))
        else:
            self.inputs = tf.identity(external_input, name="{}_inputs".format(name))
        self.targets = tf.placeholder(tf.float32, [None, 1, 1, 1], name='{}_targets'.format(name))
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        net = self.inputs
        
        #Criando camadas
        for layer in layers:
            self.layers[layer.name] = net = layer.create_layer(net, scope="disc")
            self.description += "{}".format(layer.get_description)
            
        self.logits = net
        self.final_result = tf.nn.softmax(self.logits, name="{}_output".format(name))

class Refinador:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, batch_norm=True, skip_connections=True):
        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='rconv_1_1'))
            #Aqui no meio tem uma maluquice nas conv2d
            layers.append(Conv2d(kernel_size=1, strides=[1, 1, 1, 1], output_channels=1, name='rconv_2_1'))
            
        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        net = self.inputs
        
        net = layers[0].create_layer(net, scope="ref")
        #Repeticao de layers
        with tf.variable_scope("convref", reuse=False):
            for i in range(4):
                l1 = Conv2d(kernel_size=3, strides=[1,1,1,1], output_channels=64, name="conv_rep_{}_1".format(i))
                l2 = Conv2d(kernel_size=3, strides=[1,1,1,1], output_channels=64, name="conv_rep_{}_2".format(i))
                l1 = l1.create_layer(net)
                l1 = l2.create_layer(l1)
                net = tf.nn.relu(tf.add(net, l1))
        net = layers[1].create_layer(net, scope="ref")
        with tf.variable_scope("convref", reuse=False):
            self.final_result = tf.tanh(net, name="output")
            
        self.dis = Discriminador(name="real", external_input=self.final_result)
        
class Modelo:    
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1
    
    def __init__(self):
    
        self.ref = Refinador()
        self.refiner_step = tf.Variable(0, name="refiner_step", trainable=False)
        self.discrim_step = tf.Variable(0, name="discrim_step", trainable=False)
        self.learning_rate= 0.001
        
        #UM discriminador para os exemplos reais e outro para os exemplos do refinador
        self.dis_real = Discriminador(name="real")
        
        #Computando os loss agora
        #Refinador
        se_lossr = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.ref.dis.logits, labels=tf.ones_like(self.ref.dis.logits, dtype=tf.int32)[:,:,:,0])
        self.realism_loss = tf.reduce_sum(se_lossr, [1,2], name="realism_loss")
        self.regularization_loss = 0.5 * tf.reduce_sum(tf.abs(self.ref.final_result - self.ref.inputs), [1,2,3], name="regularization_loss")
        self.refiner_loss = tf.reduce_mean(self.realism_loss + self.regularization_loss, name="refiner_loss")
        
        #Discriminador
        se_lossd = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dis_real.logits, labels=tf.zeros_like(self.dis_real.logits, dtype=tf.int32)[:,:,:,0])
        self.refiner_d_loss = tf.reduce_sum(se_lossd, [1,2], name="refiner_d_loss")
        se_losss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.ref.dis.logits, labels=tf.ones_like(self.ref.dis.logits, dtype=tf.int32)[:,:,:,0])
        self.synthetic_d_loss = tf.reduce_sum(se_losss, [1,2], name="synthetich_d_loss")
        self.discrim_loss = tf.reduce_mean(self.refiner_d_loss + self.synthetic_d_loss, name="discrim_loss")
        
        #Otimizadores
        var_list = tf.contrib.framework.get_variables("convref")
        self.refiner_optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.refiner_loss, self.refiner_step, var_list=var_list)
        var_list = tf.contrib.framework.get_variables("convdisc");
        self.discrim_optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.discrim_loss, self.discrim_step, var_list=var_list)    
        
def saveImage(test_inputs, test_target, test_result, batch_num, n = 12):
    fig, axs = plt.subplots(4, n, figsize=(n * 3, 10))
    fig.suptitle("Accuracy: {}, {}".format(0, ""), fontsize=20)
    for example_i in range(n):
        axs[0][example_i].imshow(np.reshape(test_inputs[example_i], [128,128]), cmap='gray')
        axs[1][example_i].imshow(np.reshape(test_target[example_i], [128,128]), cmap='gray')
        axs[2][example_i].imshow(np.reshape(test_result[example_i], [128,128]),cmap='gray')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = 'image_plots/gan'
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    plt.close('all')
    print("Test sample saved in {}/figure{}.jpg".format(IMAGE_PLOT_DIR, batch_num))
    return buf
        
def train():
    dataset = Dataset(batch_size=256, input_path="result/export", ground_truth="data128_128/inputs/train", test_input_path="result/inputs", test_ground_truth="data128_128/inputs/test")
    modelo = Modelo()
    
    nepoch = 3000
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("logs/gan", sess.graph)
        
        real_labels = np.reshape(np.full((dataset.batch_size,1), 1),(dataset.batch_size, 1, 1, 1))
        fake_labels = np.reshape(np.full((dataset.batch_size,1), 0),(dataset.batch_size, 1, 1, 1))
        
        #Treinando um pouco o refinador
        print("Treinando um pouco o refinador")
        refepoch = round(100/dataset.num_batches_in_epoch())+1
        for k in range(refepoch):
            dataset.reset_batch_pointer()        
            for batch_i in range(dataset.num_batches_in_epoch()):
                batch_inputs, batch_targets = dataset.next_batch()
                batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                batch_inputs = np.multiply(batch_inputs, 1.0 / 255)
                
                batch_targets = np.reshape(batch_targets,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                batch_targets = np.multiply(batch_targets, 1.0 / 255)
                
                ref_loss = sess.run([modelo.refiner_loss],feed_dict={modelo.ref.inputs: batch_inputs, modelo.ref.is_training: True})
                print("{} de {} | Refiner Loss: {}".format(k * dataset.num_batches_in_epoch() + batch_i + 1,refepoch*dataset.num_batches_in_epoch(),ref_loss))
            
        #Treinando um pouco o discriminador
        print("Treinando um pouco o discriminador")
        refepoch = round(400/dataset.num_batches_in_epoch())+1
        for k in range(refepoch):
            dataset.reset_batch_pointer()        
            for batch_i in range(dataset.num_batches_in_epoch()):
                batch_inputs, batch_targets = dataset.next_batch()
                
                batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                batch_inputs = np.multiply(batch_inputs, 1.0 / 255)
                batch_targets = np.reshape(batch_targets,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                batch_targets = np.multiply(batch_targets, 1.0 / 255)
                
                dis_loss = sess.run([modelo.discrim_loss],feed_dict={modelo.dis_real.inputs: batch_targets, modelo.dis_real.targets: real_labels, modelo.ref.inputs: batch_inputs, modelo.ref.dis.targets: fake_labels, modelo.dis_real.is_training: True})
                print("{} de {} | Discriminator Loss: {}".format(k * dataset.num_batches_in_epoch() + batch_i + 1,refepoch*dataset.num_batches_in_epoch(),dis_loss))
        
        print("Treinando os dois modelos")
        for epoch_i in range(nepoch):
            dataset.reset_batch_pointer()
            
            for batch_i in range(dataset.num_batches_in_epoch()):
                batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
                batch_inputs, batch_targets = dataset.next_batch()
                
                batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                batch_targets = np.reshape(batch_targets,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                batch_inputs = np.multiply(batch_inputs, 1.0 / 255)
                batch_targets = np.multiply(batch_targets, 1.0 / 255)
                
                ref_loss = sess.run([modelo.refiner_loss],feed_dict={modelo.ref.inputs: batch_inputs, modelo.ref.is_training: True, modelo.ref.dis.is_training: False})
                
                dis_loss = sess.run([modelo.discrim_loss],feed_dict={modelo.dis_real.inputs: batch_targets, modelo.dis_real.targets: real_labels, modelo.ref.inputs: batch_inputs, modelo.ref.dis.targets: fake_labels, modelo.dis_real.is_training: True})
                print("#{}| Disc. Loss: {} | Ref. Loss: {}".format(batch_num, dis_loss, ref_loss))
                
                if batch_num % 100 == 0:
                    #Testar e exportar o bagulho
                    test_input, test_target = dataset.test_inputs, dataset.test_outputs
                    test_input, test_target = test_input[:dataset.batch_size], test_target[:dataset.batch_size]
                    test_input = np.reshape(test_input, (dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                    test_target = np.reshape(test_target, (dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                    
                    result = sess.run([modelo.ref.final_result],feed_dict={modelo.ref.inputs: test_input, modelo.ref.is_training: False})
                    result = np.reshape(result,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                    
                    mse = np.mean(np.abs(result - test_target))
                    print("Test MSE: {}".format(mse))
                    
                    imageBuffer = saveImage(test_input, test_target, result, batch_num)
                    image = tf.image.decode_png(imageBuffer.getvalue(), channels=4)
                    image = tf.expand_dims(image, 0)
                    image_summary_op = tf.summary.image("plot", image)
                    image_summary = sess.run(image_summary_op)
                    summary_writer.add_summary(image_summary)
            
            fake_input, lista = dataset.get_hist_batch()
            fake_input = sess.run([modelo.ref.final_result],feed_dict={modelo.ref.inputs: batch_inputs, modelo.ref.dis.inputs: batch_targets, modelo.ref.is_training: False, modelo.ref.dis.is_training: False})
            dataset.set_hist_batch(fake_input, lista)
                 
        
if __name__ == '__main__':
    train()        
        
        
        
        
        
        
        
        
        
        
        
        
        
