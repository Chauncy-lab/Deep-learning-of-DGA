import logging
import os
import random as rn
import sys
import matplotlib
import numpy as np
import pandas as pd
import argparse
import logging
import os
import string
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras.optimizers import RMSprop, adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Conv1D, Dropout, concatenate, LSTM, RepeatVector, Dense, TimeDistributed, \
    LeakyReLU, BatchNormalization, AveragePooling1D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, plot_model

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
sys.path.append("../detect_DGA")

from myCode.reinforcement_learning_GAN.dga_GAN.neuralnetwork_classifier.classifier_model import Model, \
    verysmall_baseline

#from features.data_generator import load_features_dataset, load_both_datasets

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)




# def test_suppobox(X_test, y_test):
#     X_test2, y_test2 = load_features_dataset(dataset="suppobox")
#     X_test = np.concatenate((X_test, X_test2))
#     y_test = np.concatenate((y_test, y_test2))
#     return shuffle(X_test, y_test, random_state=42)

def __build_dataset(n_samples=10000, maxlen=15, validation_split=0.33):
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer() #创建对象二值化
    # df = pd.DataFrame(
    #     pd.read_csv("resources/datasets/legitdomains.txt",
    #                 sep=" ",
    #                 header=None,
    #                 names=['domain'],
    #                 ))


    # 根据标签读取数据，加载正常样本（负样本）
    all = pd.DataFrame(pd.read_csv(
        filepath_or_buffer="resources/datasets/legit_dga_domains.csv",
        sep=",",
        usecols=['domain', 'class'],
    ))
    # 根据标签读取数据，加载dga家族suppodict样本（正样本）
    suppo = pd.DataFrame(pd.read_csv(
        filepath_or_buffer="resources/datasets/suppodict.csv",
        sep=",",
        usecols=['domain', 'class']))

    # suppo = pd.DataFrame(
    #     pd.read_csv("resources/datasets/suppodict.txt",
    #                 sep=" ",
    #                 header=None,
    #                 names=['domain'],
    #                 ))

    # 合并所有样本
    df = pd.concat((all, suppo))
    print(df)
    #重新筛选，当域名长度大于5保留
    df = df.loc[df['domain'].str.len() > 5]
    #随机抽取数据集中的一部分，n为要抽取的行数，random_state为随机数发生器种子
    if n_samples:
        df = df.sample(n=n_samples, random_state=42)
    #将domian值赋予X
    X_ = df['domain'].values
    # preprocessing text（预处理文本）
    tk = Tokenizer(char_level=True)
    tk.fit_on_texts(string.ascii_lowercase + string.digits + '-' + '.')
    seq = tk.texts_to_sequences(X_)
    X = sequence.pad_sequences(seq, maxlen=maxlen)
    y = np.ravel(lb.fit_transform(df['class'].values))

    return X, y


# def both_datasets():
#     legit = pd.DataFrame(
#         pd.read_csv('/home/archeffect/PycharmProjects/adversarial_DGA/resources/datasets/all_legit.txt',
#                     header=None,
#                     index_col=False
#                     ).sample(10000)
#     )
#     y_legit = np.ravel(np.ones(len(legit), dtype=int))
#
#     generated = pd.read_csv(
#         "experiments/20171219-235309_BEST/samples.txt",
#         index_col=None, header=None)
#     y_generated = np.ravel(np.zeros(len(generated), dtype=int))
#
#     X = pd.concat((generated, legit), axis=0)
#     y = np.concatenate((y_generated, y_legit))
#     from sklearn.preprocessing import LabelBinarizer
#     lb = LabelBinarizer()
#     y = lb.fit_transform(y=y)
#     from features.features_extractors import get_feature_union
#     ft = get_feature_union(-1)
#     X = ft.fit_transform(X)
#     return X, y
#
#
# def load_domains(n_samples=None):
#     from sklearn.preprocessing import LabelBinarizer
#     lb = LabelBinarizer()
#     df = pd.DataFrame(pd.read_csv("../detect_DGA/datasets/legit_dga_domains.xlsx", sep=","))
#     if n_samples:
#         df = df.sample(n=n_samples, random_state=42)
#     X = df['domain'].values
#     y = np.ravel(lb.fit_transform(df['class'].values))
#     return X, y


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from myCode.reinforcement_learning_GAN.dga_GAN.neuralnetwork_classifier.classifier_model import pierazzi_baseline_NEW

    # model.classification_report(X_test, y_test, plot=False)
    X, y = __build_dataset(int(10000 * 1.33)) # 建立数据集合

    # print(y)
    # X = X.sample(n=1000, random_state=42)
    # # print(X)
    test_split = 0.33
    # batch_size = 32
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split) #进行数据切割
    # #定义训练次数
    # epochs = X_train.shape[0] // batch_size
    # #初始化keras模型，定义神经网络层数，并创建文件夹保存本次训练的数据记录
    # model = Model(model=pierazzi_baseline_NEW(), directory="2020test" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # #开始训练数据
    # model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=test_split)
    # model.classification_report(X_test, y_test)


    batch_size = 40
    epochs = X_train.shape[0] // batch_size
    # # for batch_size in range(10, 110, 10):
    model = Model(
        directory="neuralnetwork_classifier/saved_models/pieraz_norm_30_100")
    model.classification_report(X=X, y=y, plot=True, save=True,
                                directory="experiments/20171219-235309_BEST")

    # model = Model(model=verysmall_baseline(), directory="test_%s/verysmall_%s" % (epochs, batch_size))
    # model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=test_split, early=False)
    # model.classification_report(X, y, plot=False)
    # print(model.get_model().predict(['ronncacncoouctm']))
    # model.plot_AUC(X_test, y_test, save=True)
    pass
