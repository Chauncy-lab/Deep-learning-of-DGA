import numpy as np
from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D, Embedding, TimeDistributed, Lambda, Dropout
from keras.layers import Dense, Input
from keras.layers import LSTM, RepeatVector
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import TensorBoard

word_index = 38
embedding_length = 20
timesteps = 15

x = np.array(np.random.randint(0, 38, size=[2, 15]))
x = np.expand_dims(x, axis=2)
# x = K.expand_dims(x, axis=2)
print(x)
inp = Input((timesteps, 1))
# generator_output = TimeDistributed(Dense(word_index, activation='softmax'), name='decoder_end')(
#     inp)  # output_shape = (samples, maxlen, max_features )
manual_embedding = Dense(embedding_length, activation='linear')
embedded = TimeDistributed(manual_embedding, name='manual_embedding', trainable=False)(
    inp)  # this is actually the first layer of the discriminator in the joined GAN

model = Model(inp, embedded)
model.summary()

print(model.predict_on_batch(x))
#
# noise2 = np.random.randint(0, 38, size=[1, 15])
# enc_inputs = Input(shape=(15,), name="encoderInput")
# encoded = Embedding(word_index, embedding_length)(enc_inputs)
# model2 = Model(enc_inputs, encoded)
# model2.summary()
#
# print(noise2)
# print(model2.predict(noise2))
