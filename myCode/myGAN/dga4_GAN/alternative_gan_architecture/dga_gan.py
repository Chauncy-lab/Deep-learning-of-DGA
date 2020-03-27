import logging
import string
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

from myCode.reinforcement_learning_GAN.dga_GAN.alternative_gan_architecture.gan_model import GAN_Model, generate_dataset

logging.basicConfig()


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


class DGA_GAN(object):
    def __init__(self, batch_size):
        summary = True
        self.logger = logging.getLogger(__name__)
        self.x_train, word_index, self.inv_map = generate_dataset(n_samples=batch_size)

        self.DCGAN = GAN_Model(batch_size=batch_size, timesteps=self.x_train.shape[1], word_index=word_index,
                               summary=True)
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=2, preload_weights=None):
        self.logger.setLevel(logging.DEBUG)
        noise_input = None
        tb_gan = TensorBoard(log_dir='.logs/gan', write_graph=False)
        tb_gan.set_model(self.DCGAN.adversarial_model())
        tb_disc = TensorBoard(log_dir='.logs/disc', write_graph=False)
        tb_disc.set_model(self.DCGAN.discriminator())
        if preload_weights:
            self.adversarial.load_weights3(filepath="weights/dga_gan.h5", by_name=True)
            self.discriminator.load_weight4(filepath="weights/dga_discriminator.h5", by_name=True)
            pass
        for i in range(train_steps):
            if i > 0:
                self.logger.setLevel(logging.INFO)
            self.logger.debug("train step: %s" % i)
            # loading training set
            domains_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size // 2), :, :]
            # generating random noise
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size // 2, 128])
            self.logger.debug("noise shape:")
            self.logger.debug(noise.shape)
            # predict fake domains
            self.logger.debug("generating domains_fake...")
            domains_fake = self.generator.predict(noise)  # fake domains
            self.logger.debug("sampling fake domains....")
            domains_fake = K.eval(K.softmax(domains_fake))
            self.logger.debug("real domains shape")
            self.logger.debug(domains_train.shape)
            self.logger.debug("fake domains shape")
            self.logger.debug(domains_fake.shape)
            # concatenating fake and train domains, labeled with 0 (real) and 1 (fake)
            x = np.concatenate((domains_train, domains_fake))
            # x = np.expand_dims(x, axis=2)
            y = np.ones([batch_size, 1])  # size 2x batch size of x
            y[batch_size // 2:, :] = 0
            import tensorflow as tf
            # x = tf.convert_to_tensor(x)

            self.logger.debug("X:")
            self.logger.debug(x.shape)
            self.logger.debug("y:")
            self.logger.debug(y.shape)
            # self.logger.debug(y)
            # training discriminator
            ####
            self.logger.debug("training discriminator")
            d_loss = self.discriminator.train_on_batch(x=x, y=y)
            # self.discriminator.trainable = False
            # dataset for adversial model
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 128])  # random noise
            y = np.ones([batch_size, 1])
            # training adversial model
            a_loss = self.adversarial.train_on_batch(x=noise, y=y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            self.logger.info(log_mesg)
            if (i % 10) == 0:
                self.write_log(tb_gan, names=['loss', 'acc'], logs=a_loss, batch_no=i // 10)
                self.write_log(tb_disc, names=['loss', 'acc'], logs=d_loss, batch_no=i // 10)
                self.logger.info("saving weights...")
                self.discriminator.save_weights(filepath="weights/dga_discriminator.h5")
                self.adversarial.save_weights(filepath="weights/dga_gan.h5")
                noise = np.random.uniform(-1.0, 1.0, size=[5, 128])  # random noise
                self.generator.load_weights5(filepath='weights/dga_gan.h5', by_name=True)
                generated = self.generator.predict(noise)
                domains = K.eval(K.argmax(K.softmax(generated)))
                readable = self.to_readable_domain(domains)
                for i in range(5):
                    print("%s\t->\t%s" % (domains[i], readable[i]))

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()

    def to_readable_domain(self, decoded):
        domains = []
        for j in range(decoded.shape[0]):
            word = ""
            for i in range(decoded.shape[1]):
                if decoded[j][i] != 0:
                    word = word + self.inv_map[decoded[j][i]]
            domains.append(word)
        return domains


if __name__ == '__main__':
    import itertools

    batch_size = 1000
    mnist_dcgan = DGA_GAN(batch_size=batch_size)
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=10000, batch_size=batch_size, preload_weights=None)
    timer.elapsed_time()
