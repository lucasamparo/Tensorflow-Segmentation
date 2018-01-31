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

import numpy as np
import tensorflow as tf
import cv2

import tensorflow.contrib.slim as slim

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

    def __init__(self, layers=None, name="", external_input=None):    
        if external_input == None:
            self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],name='{}_inputs'.format(name))
        else:
            self.inputs = tf.identity(external_input, name="{}_inputs".format(name))
            
        self.targets = tf.placeholder(tf.int32, [None, 2], name='{}_targets'.format(name))
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        net = tf.reshape(self.inputs, [-1, self.IMAGE_HEIGHT*self.IMAGE_WIDTH*self.IMAGE_CHANNELS])
        #net = self.inputs
        
        with tf.variable_scope("convdisc", reuse=tf.AUTO_REUSE) as scope:
            self.logits = slim.stack(net, slim.fully_connected, [4096,1024,256,128,64,32,16,8,4,2], scope="fc")
            self.discrim_vars = tf.contrib.framework.get_variables(scope)

        self.final_result = tf.nn.softmax(logits=self.logits, name="{}_output".format(name))

class Refinador:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, batch_norm=True, skip_connections=True):            
        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        net = self.inputs
        with tf.variable_scope("convref", reuse=False) as scope:
            net = slim.conv2d(net, 64, 5, 1, scope="conv_1")
            for i in range(4):
                layer = net
                net = slim.conv2d(net, 64, 5, 1, scope="repeat_{}_1".format(i))
                net = slim.conv2d(net, 64, 5, 1, scope="repeat_{}_2".format(i))
                net = tf.nn.relu(tf.add(net, layer))
            net = slim.conv2d(net, 1, 1, 1, scope="conv_2")
            self.final_result = tf.nn.tanh(net, name="final_result")
            self.refiner_vars = tf.contrib.framework.get_variables(scope)       
            
        self.dis = Discriminador(name="fake", external_input=self.final_result)
        
class Modelo:    
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1
    
    def __init__(self):
    
        self.ref = Refinador()
        self.refiner_step = tf.Variable(0, name="refiner_step", trainable=False)
        self.discrim_step = tf.Variable(0, name="discrim_step", trainable=False)
        self.learning_rate= tf.placeholder(tf.float32, [], name="learning_rate")
        
        #UM discriminador para os exemplos reais e outro para os exemplos do refinador
        self.dis_real = Discriminador(name="real")
        
        #Computando os loss agora
        #Refinador
        se_lossr = tf.nn.softmax_cross_entropy_with_logits(logits=self.ref.dis.logits, labels=self.ref.dis.targets)
        self.realism_loss = tf.reduce_sum(se_lossr, name="realism_loss")
        self.regularization_loss = tf.reduce_sum(tf.abs(self.ref.final_result - self.ref.inputs), [1,2,3], name="regularization_loss")
        self.refiner_loss = tf.reduce_mean(self.realism_loss + (0.2 * self.regularization_loss), name="refiner_loss")
        
        #Discriminador
        se_lossd = tf.nn.softmax_cross_entropy_with_logits(logits=self.dis_real.logits, labels=self.dis_real.targets)
        self.synthetic_d_loss = tf.reduce_sum(se_lossd, name="synthetich_d_loss")
        se_losss = tf.nn.softmax_cross_entropy_with_logits(logits=self.ref.dis.logits, labels=self.ref.dis.targets)
        self.refiner_d_loss = tf.reduce_sum(se_losss, name="refiner_d_loss")
        self.discrim_loss = tf.reduce_mean(self.synthetic_d_loss + self.refiner_d_loss, name="discrim_loss")
        
        #Otimizadores
        var_list = self.ref.refiner_vars
        self.refiner_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.refiner_loss, self.refiner_step, var_list=var_list)
        self.refiner_only_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.regularization_loss, var_list=var_list)
        
        var_list = self.dis_real.discrim_vars
        self.discrim_only_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.synthetic_d_loss)
        self.discrim_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.discrim_loss, self.discrim_step)    
        
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
    input_path = "result/export"
    ground_truth = "data128_128/inputs/train"
    test_input_path = "result/inputs"
    test_ground_truth = "data128_128/inputs/test"
    dataset = Dataset(batch_size=32, input_path=input_path, ground_truth=ground_truth, test_input_path=test_input_path, test_ground_truth=test_ground_truth)
    modelo = Modelo()
    
    nepoch = 3000
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("logs/gan", sess.graph)
        
        _real_labels = np.reshape(np.tile(np.array([0,1], dtype=np.int32), 256), [-1,2])
        _fake_labels = np.reshape(np.tile(np.array([1,0], dtype=np.int32), 256), [-1,2])
        
        #Treinando um pouco o refinador
        print("Treinando um pouco o refinador")
        dataset.batch_size = 64
        dataset.reset_batch_pointer()
        rep = 5000
        for batch_i in range(rep):
            if batch_i % dataset.num_batches_in_epoch() == 0:
                dataset.reset_batch_pointer()
            batch_inputs, batch_targets = dataset.next_batch()
            batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
            batch_inputs = np.multiply(batch_inputs, 1.0 / 255)
            
            batch_targets = np.reshape(batch_targets,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
            batch_targets = np.multiply(batch_targets, 1.0 / 255)
            
            ref_loss, _ = sess.run([modelo.regularization_loss, modelo.refiner_only_optim],feed_dict={modelo.ref.inputs: batch_inputs, modelo.learning_rate: 1e-4, modelo.ref.is_training: True})
            print("{} de {} | Refiner (Reg.) Loss: {}".format(batch_i+1,rep,np.mean(ref_loss)))
            if (batch_i+1) % 10 == 0:
                test_input, test_target = dataset.test_inputs, dataset.test_outputs
                test_input, test_target = test_input[:dataset.batch_size], test_target[:dataset.batch_size]
                test_input = np.reshape(test_input, (dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                test_target = np.reshape(test_target, (dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                
                result, loss = sess.run([modelo.ref.final_result, modelo.regularization_loss], feed_dict={modelo.ref.inputs: test_input})

                imageBuffer = saveImage(test_input, test_target, result, batch_i+1)
                
        
        #Treinando um pouco o discriminador
        """print("Treinando um pouco o discriminador")
        dataset.batch_size = 256
        dataset.reset_batch_pointer()
        rep = 1000
        real_labels = _real_labels[:dataset.batch_size]
        fake_labels = _fake_labels[:dataset.batch_size]
        for batch_i in range(rep):
            if batch_i % dataset.num_batches_in_epoch() == 0:
                dataset.reset_batch_pointer()
            
            batch_inputs, batch_targets = dataset.next_batch()
            
            batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
            batch_inputs = np.multiply(batch_inputs, 1.0 / 255)
            batch_targets = np.reshape(batch_targets,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
            batch_targets = np.multiply(batch_targets, 1.0 / 255)
            
            batch_ = np.vstack([batch_inputs, batch_targets])
            label_ = np.vstack([real_labels, fake_labels])
            
            permutation = np.random.permutation(len(batch_))
            batch = [batch_[i] for i in permutation]
            label = [label_[i] for i in permutation]
            
            batch_inputs = np.array(batch)[:dataset.batch_size]
            label_inputs = np.array(label, dtype=np.int32)[:dataset.batch_size]
            
            dis_loss, _ = sess.run([modelo.synthetic_d_loss, modelo.discrim_only_optim],feed_dict={modelo.dis_real.inputs: batch_inputs, modelo.learning_rate: 1e-4, modelo.dis_real.targets: label_inputs, modelo.dis_real.is_training: True})
            print("{} de {} | Discriminator Loss: {}".format(batch_i,rep,dis_loss))
                
            if batch_i % 100 == 0:
                test_input, test_target = dataset.test_inputs, dataset.test_outputs
                test_input, test_target = test_input[:dataset.batch_size], test_target[:dataset.batch_size]
                test_input = np.reshape(test_input, (dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                test_target = np.reshape(test_target, (dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                
                batch_ = np.vstack([test_input, test_target])
                label_ = np.vstack([real_labels, fake_labels])
                
                permutation = np.random.permutation(len(batch_))
                batch = [batch_[i] for i in permutation]
                label = [label_[i] for i in permutation]
                
                result_real = sess.run([modelo.dis_real.final_result],feed_dict={modelo.dis_real.inputs: batch, modelo.dis_real.is_training: False})
                
                test_error = np.abs(result_real[0] - label)
                print(result_real[0][5], label[5])
                print(result_real[0][10], label[10])
                print(result_real[0][15], label[15])
                print(result_real[0][20], label[20]) 
                print("Test Error > SUM:{} MEAN:{}".format(np.sum(test_error),np.mean(test_error)))"""
        
        print("Treinando os dois modelos")
        dataset.batch_size = 32
        real_labels = _real_labels[:dataset.batch_size]
        fake_labels = _fake_labels[:dataset.batch_size]
        for epoch_i in range(nepoch):
            dataset.reset_batch_pointer()
            
            for batch_i in range(dataset.num_batches_in_epoch()):
                batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
                batch_inputs, batch_targets = dataset.next_batch()
                
                batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                batch_targets = np.reshape(batch_targets,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                batch_inputs = np.multiply(batch_inputs, 1.0 / 255)
                batch_targets = np.multiply(batch_targets, 1.0 / 255)
                
                ref_loss, _ = sess.run([modelo.refiner_loss, modelo.refiner_optim],feed_dict={modelo.ref.inputs: batch_inputs,  modelo.ref.dis.targets: real_labels, modelo.learning_rate: 1e-4, modelo.ref.is_training: True, modelo.ref.dis.is_training: False})
                
                dis_loss, _ = sess.run([modelo.discrim_loss, modelo.discrim_optim],feed_dict={modelo.dis_real.inputs: batch_targets, modelo.dis_real.targets: real_labels, modelo.ref.inputs: batch_inputs, modelo.ref.dis.targets: fake_labels, modelo.learning_rate: 1e-6, modelo.dis_real.is_training: True})
                print("#{}| Disc. Loss: {} | Ref. Loss: {}".format(batch_num, dis_loss, ref_loss))
                
                if batch_num % 100 == 0:
                    #Testar e exportar o bagulho
                    test_input, test_target = dataset.test_inputs, dataset.test_outputs
                    test_input, test_target = test_input[:dataset.batch_size], test_target[:dataset.batch_size]
                    test_input = np.reshape(test_input, (dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                    test_target = np.reshape(test_target, (dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                    
                    result = sess.run([modelo.ref.final_result],feed_dict={modelo.ref.inputs: test_input, modelo.ref.is_training: False})
                    #result = np.reshape(result,(dataset.batch_size, modelo.IMAGE_HEIGHT, modelo.IMAGE_WIDTH, 1))
                    
                    mse = np.mean(np.abs(result[0] - test_target))
                    print("Test MSE: {}".format(mse))
                    
                    imageBuffer = saveImage(test_input, test_target, result[0], batch_num)
                    image = tf.image.decode_png(imageBuffer.getvalue(), channels=4)
                    image = tf.expand_dims(image, 0)
                    image_summary_op = tf.summary.image("plot", image)
                    image_summary = sess.run(image_summary_op)
                    summary_writer.add_summary(image_summary)
            
                fake_input, lista = dataset.get_hist_batch()
                fake_input = sess.run([modelo.ref.final_result],feed_dict={modelo.ref.inputs: batch_inputs, modelo.ref.is_training: False})
                dataset.set_hist_batch(fake_input, lista)
                 
        
if __name__ == '__main__':
    train()        
