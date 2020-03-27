# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import pickle
import gzip
import numpy as np
import tflearn
from sklearn import metrics
import tflearn.datasets.mnist as mnist




X, Y, testX, testY = mnist.load_data(one_hot=True) #导入数据

def do_DNN(X, Y, testX, testY):
    # Building deep neural network
    input_layer = tflearn.input_data(shape=[None, 784]) #定义输入参数形状，只有一个维度
    dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001) #连接全网络
    dropout1 = tflearn.dropout(dense1, 0.8) #防止过拟合
    dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout2 = tflearn.dropout(dense2, 0.8)
    softmax = tflearn.fully_connected(dropout2, 10, activation='softmax') #定义输出层，使用softmax分类

    # Regression using SGD with learning rate decay and Top-3 accuracy
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000) #梯度下降SGD
    top_k = tflearn.metrics.Top_k(3) #定义，真实结果在预测结果前3中就算正确
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),
              show_metric=True, run_id="dense_model")

def do_rnn(X, Y, testX, testY):
    X = np.reshape(X, (-1, 28, 28))  #将训练集转换28x28的矩阵
    testX = np.reshape(testX, (-1, 28, 28)) #将测试集转换28x28的矩阵

    net = tflearn.input_data(shape=[None, 28, 28]) #定义输入参数形状，28x28
    net = tflearn.lstm(net, 128, return_seq=True)
    net = tflearn.lstm(net, 128)
    net = tflearn.fully_connected(net, 10, activation='softmax') #fully_connected 是指前一层的每一个神经元都和后一层的所有神经元相连
    net = tflearn.regression(net, optimizer='adam', #最后应用一个分类器，定义优化器，学习率，损失函数
                         loss='categorical_crossentropy', name="output1")
    model = tflearn.DNN(net, tensorboard_verbose=2) #创建神经网络实体
    model.fit(X, Y, n_epoch=1, validation_set=(testX,testY), show_metric=True,
          snapshot_step=100) #n_epoch：整个训练的次数，validation_set：验证数据集的比例，show_metric是否展示训练过程，snapshot_step训练步长


# do_DNN(X, Y, testX, testY)
do_rnn(X, Y, testX, testY)