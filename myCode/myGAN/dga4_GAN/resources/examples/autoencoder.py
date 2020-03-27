import string

import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D, Embedding, TimeDistributed, Lambda
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import LSTM, RepeatVector
from keras.layers.merge import concatenate
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop
from keras.models import Sequential

# fix random seed for reproducibility
np.random.seed(7)
# np.set_printoptions(linewidth=2000, threshold=100000)
# load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
n_samples = 256


# loading db
# lb = LabelBinarizer()
def generate_dataset(maxlen=15):
    df = pd.DataFrame(pd.read_csv("../dataset/legitdomains.txt", sep=" ", header=None, names=['domain']))
    if n_samples:
        df = df.sample(n=n_samples, random_state=42)
    X_ = df['domain'].values
    # y = np.ravel(lb.fit_transform(df['class'].values))

    # preprocessing text
    tk = Tokenizer(char_level=True)
    tk.fit_on_texts(string.lowercase + string.digits + '-' + '.')
    print("word index: %s" % len(tk.word_index))
    seq = tk.texts_to_sequences(X_)
    # for x, s in zip(X_, seq):
    #     print(x, s)
    # print("")
    X = sequence.pad_sequences(seq, maxlen=maxlen)
    print("X shape after padding: " + str(X.shape))
    # print(X)
    inv_map = {v: k for k, v in tk.word_index.iteritems()}

    return X, len(tk.word_index), inv_map


class Autoencoder(object):
    def __init__(self, X, word_index_len):
        # autoencoder params
        # self.X = X
        # self.y = y
        self.timesteps = 15  # lunghezza vettore
        self.filters = [20, 10]
        self.kernels = [2, 3]
        self.embedding_length = 20  # lunghezza vettore embedded
        self.word_index = word_index_len

        self.A = None
        self.D = None
        self.E = None

    def encoder(self):
        if self.E:
            return self.E

        enc_convs = []
        enc_inputs = Input(shape=(self.timesteps,), name="encoderInput")
        encoded = Embedding(self.word_index, self.embedding_length, input_length=self.timesteps)(enc_inputs)
        for i in range(2):
            conv = Conv1D(self.filters[i],
                          self.kernels[i],
                          padding='same',
                          activation='relu',
                          strides=1)(encoded)
            conv = Dropout(0.1)(conv)
            conv = MaxPooling1D()(conv)
            enc_convs.append(conv)

        encoded = concatenate(enc_convs)
        encoded = LSTM(128, name="encoderLSTM")(encoded)
        self.E = Model(inputs=enc_inputs, outputs=encoded, name='encoder')
        plot_model(self.E, to_file="encoder.png", show_shapes=True)
        return self.E

    def decoder(self):
        if self.D:
            return self.D

        dec_inputs = Input(shape=(128,))
        decoded = RepeatVector(self.timesteps)(dec_inputs)
        decoded = LSTM(128, return_sequences=True)(decoded)
        dec_convs = []
        for i in range(2):
            conv = Conv1D(self.filters[i],
                          self.kernels[i],
                          padding='same',
                          activation='relu',
                          strides=1)(decoded)
            conv = Dropout(0.1)(conv)
            # conv = MaxPooling1D()(conv)
            dec_convs.append(conv)

        decoded = concatenate(dec_convs)
        emb = Dense(self.word_index, activation='sigmoid')
        decoded = TimeDistributed(emb, name="timedistr")(decoded)
        self.D = Model(inputs=dec_inputs, outputs=decoded, name='decoder')
        plot_model(self.D, to_file="decoder.png", show_shapes=True)
        return self.D

    def autoencoder(self):
        if self.A:
            return self.A

        self.A = Sequential(name='autoencoder')
        self.A.add(self.encoder())
        self.A.add(self.decoder())
        # self.A.add(Sampling(output_dim=self.timesteps, trainable=False))
        self.A.add(Lambda(K.softmax, output_shape=(self.timesteps, self.word_index)))
        self.A.add(Lambda(K.argmax, output_shape=(self.timesteps,)))
        self.A.add(Lambda(K.cast, output_shape=(self.timesteps,), arguments={'dtype': 'float'}))
        # self.A.add(Lambda(lambda_sampling,output_shape=(self.timesteps,)))
        # self.A.add(Sampling(output_dim=(self.timesteps)))
        optimizer = RMSprop(lr=0.01)
        self.A.compile(loss='mse', optimizer='rmsprop')
        self.A.summary()
        return self.A

    def predict_(self, X, inv_map):
        aut = self.autoencoder()
        preds = aut.predict(x=X, verbose=2)
        domains = []
        for j in range(preds.shape[0]):
            word = ""
            for i in range(preds.shape[1]):
                k = self.sample(preds[j][i])
                if k > 0:
                    word = word + inv_map[k]
            domains.append(word)

        domains = np.char.array(domains)
        return domains

        # def sample(self, preds, temperature=1.0):
        #     # helper function to sample an index from a probability array
        #     preds = np.asarray(preds).astype('float32')
        #     preds = np.log(preds) / temperature
        #     exp_preds = np.exp(preds)
        #     preds = exp_preds / np.sum(exp_preds)
        #     probas = np.random.multinomial(1, preds, 1)
        #     return np.argmax(probas)


def lambda_sampling(x):
    temperature = 1.0
    x = K.log(x) / temperature
    exp_preds = K.exp(x)
    x = exp_preds / K.sum(exp_preds)
    x = K.argmax(x, axis=2)
    return K.cast(x, 'float')


if __name__ == '__main__':
    #   from detect_DGA import MyClassifier

    timesteps = 15
    batch_size = 10
    x_train, word_index, inv_map = generate_dataset(timesteps)
    print(x_train.shape)
    x_train = x_train.astype(float)
    print(x_train.dtype)
    aenc = Autoencoder(x_train, word_index)
    # out = aenc.autoencoder().train_on_batch(x_train, x_train)

    # for out in aenc.encoder().layers:
    #     print(out)
    pred = aenc.autoencoder().predict(x_train)
    print(pred.shape)
    print(pred)
    # aenc.autoencoder().fit(x_train, x_train,
    #                        epochs=100,
    #                        batch_size=128,
    #                        shuffle=True,
    #                        validation_split=0.33,
    #                        # validation_data=(x_test_noisy, x_test),
    #                        callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)]
    #                        )


    #####################################
    # print("ENCODER DEMO")
    # encoded = aenc.encoder().predict_on_batch(X)
    # print(encoded)
    # # domains = aenc.predict_(X, inv_map)
    # print("DECODER DEMO")
    # decoded = aenc.decoder().predict_on_batch(encoded)
    # print(decoded)
    # print("SAMPLING DECODED DOMAINS")
    # domains=[]
    # for j in range(decoded.shape[0]):
    #     word = ""
    #     for i in range(decoded.shape[1]):
    #         k = aenc.sample(decoded[j][i])
    #         if k > 0:
    #             word = word + inv_map[k]
    #     domains.append(word)
    #
    # domains = np.char.array(domains)
    # print(domains)
    #
    # # print("TESTING DOMAINS")
    # # rndf = MyClassifier(directory="/home/archeffect/PycharmProjects/detect_DGA/models/RandomForest tra:sup tst:sup")
    # # true = np.ravel(np.zeros(len(domains), dtype=int))
    # # res = rndf.predict(domains, true, verbose=False)
