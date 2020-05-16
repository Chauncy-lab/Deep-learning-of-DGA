import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Lambda, Dense
from keras.models import Model


def lambda_sampling(x):
    temperature = 1.0
    x = K.log(x) / temperature
    exp_preds = K.exp(x)
    x = exp_preds / K.sum(exp_preds)
    x = K.argmax(x, axis=2)
    return K.cast(x, dtype='float32')


def sampling_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    return tuple(shape[:2])


class Sampling(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Sampling, self).__init__(**kwargs)

    def build(self, input_shape):
        # print(input_shape)

        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape, self.output_dim),
                                      initializer='uniform',
                                      trainable=False)
        super(Sampling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        temperature = 1.0
        x = K.log(x) / temperature
        exp_preds = K.exp(x)
        x = exp_preds / K.sum(exp_preds)
        x = K.argmax(x, axis=2)
        x = K.cast(x, 'float')
        return x

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 3  # only valid for 3D tensors
        return tuple(shape[:2])


if __name__ == '__main__':
    from gan_model import GAN_Model
    from dga_gan import DGA_GAN

    batch_size = 10
    timesteps = 15

    # dga = DGA_GAN(batch_size=batch_size)
    X, word_index = DGA_GAN(batch_size=batch_size).build_dataset(n_samples=batch_size)
    # gan = GAN_Model(batch_size=batch_size, timesteps=timesteps, word_index=word_index)
    ########### MODEL
    X = X.astype(float)
    # print(X.dtype)
    inp = Input(shape=(timesteps, word_index))
    exa = Lambda(lambda_sampling, output_shape=sampling_output_shape, name='my_sampling', trainable=False)(inp)

    model = Model(inputs=inp, outputs=exa)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    noise = np.random.uniform(0, 1, size=(batch_size, timesteps, 38))

    print("NOISE INPUT")
    print(noise.shape)
    print(noise.dtype)
    y = np.ones([batch_size, 1])  # size 2x batch size of x

    sampling = model.predict_on_batch(noise)
    loss = model.fit(x=noise, y=noise, verbose=1)
    print("SAMPLED OUTPUT")
    print(sampling.shape)
    print(sampling)
    #################



    #############  decoding into readable domain
    # tk = Tokenizer(char_level=True)
    # tk.fit_on_texts(string.lowercase + string.digits + '-' + '.')
    # inv_map = {v: k for k, v in tk.word_index.iteritems()}
    #
    # domains = []
    #
    # for j in range(decoded.shape[0]):
    #     word = ""
    #     for i in range(decoded.shape[1]):
    #         if decoded[j][i] != 0:
    #             word = word + inv_map[decoded[j][i]]
    #     domains.append(word)
    #
    # domains = np.char.array(domains)
    # for domain in domains:
    #     print(domain)
