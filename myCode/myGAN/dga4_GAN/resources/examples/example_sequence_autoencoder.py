import string

import keras.backend as K
import numpy as np
import pandas as pd
from keras import Input
from keras.callbacks import TensorBoard
from keras.layers import Conv1D, Dropout, MaxPooling1D, concatenate, LSTM, RepeatVector, Dense, TimeDistributed, Lambda
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # probas = np.random.multinomial(1, preds, 1)
    return np.argmax(preds)


def lambda_sampling(x):
    temperature = 1.0
    x = K.log(x) / temperature
    exp_preds = K.exp(x)
    x = exp_preds / K.sum(exp_preds)
    x = K.argmax(x, axis=2)
    x = K.expand_dims(x, axis=2)
    return K.cast(x, dtype='float32')


def generate_dataset(n_samples=None, maxlen=15):
    df = pd.DataFrame(
        pd.read_csv("/home/archeffect/PycharmProjects/adversarial_DGA/dataset/legitdomains.txt", sep=" ", header=None,
                    names=['domain']))
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


timesteps = 15
word_index = 38
embedding_length = 20
filters = [20, 10]
kernels = [2, 3]
latent_dim = 128

## ENCODER
enc_convs = []
enc_inputs = Input(shape=(15, 1), name="encoderInput")
manual_embedding = Dense(20, activation='linear', name="manual_embedding")
encoded = TimeDistributed(manual_embedding, trainable=False, name="timedistributed_embedding")(enc_inputs)

# encoded = Embedding(word_index, embedding_length, input_length=timesteps)(enc_inputs)
for i in range(2):
    conv = Conv1D(filters[i],
                  kernels[i],
                  padding='same',
                  activation='relu',
                  strides=1)(encoded)
    conv = Dropout(0.1)(conv)
    conv = MaxPooling1D()(conv)
    enc_convs.append(conv)

encoded = concatenate(enc_convs)
encoded = LSTM(128, name="encoderLSTM")(encoded)
##########
encoder = Model(enc_inputs, encoded)
## DECODER
decoded = RepeatVector(15)(encoded)
decoded = LSTM(128, return_sequences=True)(decoded)
dec_convs = []
for i in range(2):
    conv = Conv1D(filters[i],
                  kernels[i],
                  padding='same',
                  activation='relu',
                  strides=1)(decoded)
    conv = Dropout(0.1)(conv)
    # conv = MaxPooling1D()(conv)
    dec_convs.append(conv)

decoded = concatenate(dec_convs)
emb = Dense(38, activation='sigmoid')
decoded = TimeDistributed(emb, name="timedistr")(decoded)
decoded = Lambda(K.softmax)(decoded)
decoded = Lambda(K.argmax, output_shape=(timesteps,), arguments={'axis': 2}, name='argmax')(decoded)
# decoded = Lambda(K.cast, arguments={'dtype': "int32"}, name='cast')(decoded)
#########


sequence_autoencoder = Model(enc_inputs, decoded)
from keras.utils import plot_model

plot_model(sequence_autoencoder, to_file="images/sequence_autoencoder.png", show_shapes=True, show_layer_names=True)
sequence_autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')
sequence_autoencoder.summary()
noise = np.random.randint(0, 38, size=(10000, 15, 38))
# X, a, b = generate_dataset(n_samples=1)
# print(noise)
X = np.expand_dims(noise, axis=2)
# loss = sequence_autoencoder.fit(X, X, epochs=1000, batch_size=256, verbose=2, callbacks=[
#     TensorBoard(log_dir='/tmp/tb', write_graph=False)
# ], validation_split=0.33)
print(X)
print(sequence_autoencoder.predict_on_batch(X))
