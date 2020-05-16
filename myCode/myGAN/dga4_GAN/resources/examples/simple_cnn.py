# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
n_samples = 15000


def load_domains(n_samples=None):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    df = pd.DataFrame(pd.read_csv("../../detect_DGA/datasets/legit_dga_domains.csv", sep=","))
    if n_samples:
        df = df.sample(n=n_samples, random_state=42)
    X = df['domain'].values
    y = np.ravel(lb.fit_transform(df['class'].values))
    return X, y


from sklearn.model_selection import train_test_split

X, y = load_domains(n_samples=n_samples)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_matrix(X)
print(X.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(x_train)
# # truncate and pad input sequences
# max_review_length = 500
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
embedding_layer = Embedding(n_samples, embedding_vecor_length, input_length=X.shape[1], trainable=False)
MAX_SEQUENCE_LENGTH = X.shape[1]
print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(20, 2, activation='relu')(embedded_sequences)
x = MaxPooling1D(2)(x)
x = Conv1D(10, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='softmax')(x)

model = Model(sequence_input, preds)

# model = Sequential()
# model.add(Embedding(n_samples, embedding_vecor_length, input_length=X.shape[1]))
# model.add(Conv1D(filters=20, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.33)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
