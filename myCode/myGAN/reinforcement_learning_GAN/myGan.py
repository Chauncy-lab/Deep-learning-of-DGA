# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop

from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Reshape
from keras.optimizers import SGD
from PIL import Image
import argparse
import math
import matplotlib.mlab as MLA
from scipy.stats import norm

mu, sigma=(0,1)

#真实样本满足正态分布 平均值维0 方差为1 样本维度200
def x_sample(size=200,batch_size=32):
    x=[]
    for _ in range(batch_size):
        x.append(np.random.normal(mu, sigma, size))
    b = np.array(x)
    return b

#噪声样本 噪声维度维200 满足均匀分布
def z_sample(size=200,batch_size=32):
    z=[]
    for _ in range(batch_size):
        z.append(np.random.uniform(-1, 1, size))
    return np.array(z)


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=200, units=256)) ##输入维度200，输出维度为256（隐藏层结点数）.units参数使一个正整数，表示输出的维度
    model.add(Activation('relu'))#激活函数
    model.add(Dense(200))#输出维度为200（隐藏层结点数）
    model.add(Activation('sigmoid'))#激活函数
    plot_model(model, show_shapes=True, to_file='gan/keras-gan-generator_model.png')
    return model


def discriminator_model():
    model = Sequential()
    model.add(Reshape((200,), input_shape=(200,)))#输入一个一阶、拥有200个元素的一维数组，并且用reshape将输入shape转换为特定的shape
    model.add(Dense(units=256))#输出维度为256
    model.add(Activation('relu'))#激活函数
    model.add(Dense(1))#输出维度为1
    model.add(Activation('sigmoid'))
    plot_model(model, show_shapes=True, to_file='gan/keras-gan-discriminator_model.png')
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False #将自动更新设置为false
    model.add(d)
    plot_model(model, show_shapes=True, to_file='gan/keras-gan-gan_model.png')
    return model

def show_image(s):
    count, bins, ignored = plt.hist(s, 5, density=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()

def save_image(s,filename):
    count, bins, ignored = plt.hist(s, bins=20, density=True,facecolor='w',edgecolor='b') #描绘直方图
    y = norm.pdf(bins, mu, sigma)#每个bins自变量对应的y值
    l = plt.plot(bins, y, 'g--', linewidth=2)   #描绘线条
    #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.savefig(filename)#保存图片



def show_init():
    x=x_sample(batch_size=1)[0] #定义真实样本生成函数
    save_image(x,"gan/x-0.png")
    z=z_sample(batch_size=1)[0]#定义噪音源样本生成函数
    save_image(z, "gan/z-0.png")


def save_loss(d_loss_list,g_loss_list):

    plt.subplot(2, 1, 1)  # 面板设置成2行1列，并取第一个（顺时针编号）
    plt.plot(d_loss_list, 'yo-')  # 画图，染色
    #plt.title('A tale of 2 subplots')
    plt.ylabel('d_loss')

    plt.subplot(2, 1, 2)  # 面板设置成2行1列，并取第二个（顺时针编号）
    plt.plot(g_loss_list,'r.-')  # 画图，染色
    #plt.xlabel('time (s)')
    plt.ylabel('g_loss')


    plt.savefig("gan/loss.png")

if __name__ == '__main__':

    show_init()
    d_loss_list=[]
    g_loss_list = []


    """
    初始化模型
    """
    d = discriminator_model() #初始化模型
    g = generator_model() #初始化模型
    d_on_g = generator_containing_discriminator(g, d) #初始化模型
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim) #loss: 目标函数，也叫损失函数，是网络中的性能函数.optimizer:优化器，如Adam
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    """
    训练数据
    """
    batch_size = 128
    for epoch in range(500):
        print("Epoch is", epoch)

        """第一阶段"""
        noise=z_sample(batch_size=batch_size) ##生成噪音样本
        image_batch=x_sample(batch_size=batch_size)##生成真实样本
        generated_images = g.predict(noise, verbose=0)#为输入样本生成输出预测
        x= np.concatenate((image_batch, generated_images)) #噪音合并真实样本
        y=[1]*batch_size+[0]*batch_size#标上0和1得标签
        d_loss = d.train_on_batch(x, y) #对判别器进行训练，返回标量训练损失值
        print("d_loss : %f" % (d_loss))

        """第二阶段"""
        noise = z_sample(batch_size=batch_size)#生成噪音样本
        d.trainable = False #设置判别器不会更新参数
        g_loss = d_on_g.train_on_batch(noise, [1]*batch_size)#noise为噪音样本，并且使用设置“真”标签，企图欺骗对抗模型
        d.trainable = True # 训练之后又改为可更新参数
        print("g_loss : %f" % (g_loss))  #返回对抗模型标量训练损失值
        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss)

        if epoch % 100 == 1:
            # 测试阶段
            noise = z_sample(batch_size=1)
            generated_images = g.predict(noise, verbose=0)
            # print generated_images
            save_image(generated_images[0], "gan/z-{}.png".format(epoch))

    save_loss(d_loss_list, g_loss_list)
