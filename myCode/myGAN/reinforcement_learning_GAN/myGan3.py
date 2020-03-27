from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
# from utils.args_utils import *
# from utils.image_utils import *
from code.reinforcement_learning_GAN.myGan2 import combine_images

"""
《生成对抗网络入门指南》中的DCGAN
"""

def load_data():
    data = np.load('../../data/mnist.npz')
    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    return (x_train, y_train), (x_test, y_test)

class DCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.lr = 0.0002
        self.beta = 0.5
        optimizer = Adam(self.lr, self.beta)
        # 构建判别器
        self.discriminator = self.build_discriminator()
        # 编译判别器，如果这里指定了metrics，例如metrics=['accuracy']，后面train_on_batch时就会有两列，一个是loss，一个是accuracy
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        # 构建生成器
        self.generator = self.build_generator()
        #以噪声为输入，产生img
        # 输入随机噪音源
        z = Input(shape=(self.latent_dim,))
        # 根据生成器的规则生成噪音图片
        img = self.generator(z)
        # 固定判别器
        self.discriminator.trainable = False
        # 判别图像（鉴别器将生成的图像作为输入并确定有效性）
        valid = self.discriminator(img)
        # combine模型包含固定住的判别器和生成器
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, (3, 3), padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, (3, 3), strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128):
        # 这里的epochs实际上是一个batch
        d_losses = []
        g_losses = []
        if not os.path.exists("class_fonts_samples/"):
            os.mkdir("class_fonts_samples/")
        (X_train, _), (_, _) = load_data()

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)
        # 设置标签，valid表示真实数据，fake表示随机噪声
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            if epoch % 200 == 0:
                image = combine_images(gen_imgs)
                image = image * 127.5 + 127.5
                # 使用PIL库的Image对象从合并好的image（ndarray）中生成图像
                Image.fromarray(image.astype(np.uint8)).save("class_fonts_samples/" + str(epoch) + ".png")
            # 用真实图像训练一次判别器
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            # 再用生成图像训练一次判别器
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # 训练生成器
            g_loss = self.combined.train_on_batch(noise, valid)
            d_losses.append(d_loss)
            g_losses.append(g_loss)
        return d_losses, g_losses


if __name__ == "__main__":
    # args = get_args()
        dcgan = DCGAN()
    # if args.mode == "train":
        d_losses, g_losses = dcgan.train(epochs=4000, batch_size=128)
        # print(d_losses)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(d_losses, label='d_loss')
        ax.plot(g_losses, label='g_loss')
        ax.legend()
        plt.show()
    # elif args.mode == "generate":
    #     generate(BATCH_SIZE=args.batch_size, nice=args.nice)