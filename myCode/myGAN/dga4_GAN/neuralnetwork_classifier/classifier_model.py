from __future__ import print_function

import json
import logging
import os
import random as rn
import sys

from matplotlib import pyplot as plt

sys.path.append("../detect_DGA")

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential, model_from_json, save_model
from keras.utils import plot_model
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

# from plot_module import plot_classification_report

from sklearn.metrics import classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)

tf.set_random_seed(1234)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


class Model:
    #传进kreas定义好的modle，和文件夹名称：2020test+年月日时分秒
    def __init__(self, model=None, directory=None):
        self.model = model
        self.directory = directory
        # 如果不存在该文件夹则创建一个
        if not os.path.exists(self.directory):
            self.directory = os.path.join("neuralnetwork_classifier/saved_models", directory)
            # crea la cartella（创建文件夹）
            os.makedirs(self.directory)
        #进行日志输出
        self.init_logger()
        # 如果没有模型，则__load_model加载默认模型
        if not self.model:
            self.model = self.__load_model()

    def init_logger(self):
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger(__name__)
        hdlr = logging.FileHandler(os.path.join(self.directory, 'results.log'))
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

    #
    # def __call__(self, *args, **kwargs):
    #     return self.model
    #
    # def __make_exp_dir(self, directory):
    #     if not os.path.exists(directory):
    #         directory = os.path.join("saved models", time.strftime("%c"))
    #         os.makedirs(directory)
    #
    #     if socket.gethostname() == "classificatoredga":
    #         directory = "kula_" + directory
    #     return directory

    '''自己加的'''
    def plot_classification_report( title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

        report_str = '''           precision    recall  f1-score   support
                   0       1.00      1.00      1.00      4389
            accuracy                           1.00      4389
           macro avg       1.00      1.00      1.00      4389
        weighted avg       1.00      1.00      1.00      4389
        '''
        lines = report_str.split('\n')

        classes = []
        plotMat = []
        for line in lines[2: (len(lines) - 3)]:
            # print(line)
            t = line.split()
            # print(t)
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            print(v)
            plotMat.append(v)

        if with_avg_total:
            aveTotal = lines[len(lines) - 1].split()
            classes.append('avg/total')
            vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
            plotMat.append(vAveTotal)

        plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        x_tick_marks = np.arange(3)
        y_tick_marks = np.arange(len(classes))
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('Classes')
        plt.xlabel('Measures')

    def __save_model(self):
        # saving model（保存模型）
        # 保存模型josn数据参数
        json_model = self.model.to_json()
        dirmod = os.path.join(self.directory, 'model_architecture.json')
        open(dirmod, 'w').write(json_model)
        self.logger.info("model saved to %s" % dirmod)

        # saving weights（保存权重）
        dirwe = os.path.join(self.directory, 'model_weights.h5')
        self.model.save_weights(dirwe, overwrite=True)
        self.logger.info("model weights saved to %s" % dirwe)
        # 将该模型以图片形式输出并且保存（计算图）
        dirplo = os.path.join(self.directory, "model.png")
        plot_model(self.model, to_file=dirplo, show_layer_names=True,
                   show_shapes=True)
        self.logger.info("network diagram saved to %s " % dirplo)

    #加载默认模型
    def __load_model(self):
        # loading model
        try:
            model = model_from_json(open(os.path.join(self.directory, 'model_architecture.json')).read())
            model.load_weights(os.path.join(self.directory, 'model_weights.h5'))
            model.compile(loss='binary_crossentropy', optimizer='adam')
            self.logger.debug("Model %s loaded" % self.directory)
            return model
        except IOError as e:
            self.logger.error(e)

    def print_results(self, results, to_console=False):
        for key, value in sorted(results.iteritems()):
            if not "time" in key:
                foo = "%s: %.2f%% (%.2f%%)" % (key, value.mean() * 100, value.std() * 100)
                if to_console:
                    print(foo)

                self.logger.info(foo)
            else:
                foo = "%s: %.2fs (%.2f)s" % (key, value.mean(), value.std())
                if to_console:
                    print(foo)

                self.logger.info(foo)

    def save_results(self, results):
        self.print_results(results)
        _res = {k: v.tolist() for k, v in results.items()}
        with open(os.path.join(self.directory, 'data.json'), 'w') as fp:
            try:
                json.dump(_res, fp, sort_keys=True, indent=4)
            except BaseException as e:
                self.logger.error(e)

    def load_results(self):
        with open(os.path.join(self.directory, "data.json"), 'rb') as fd:
            results = json.load(fd)

        results = {k: np.asfarray(v) for k, v in results.iteritems()}
        self.print_results(results, to_console=True)
        return results

    def get_model(self):
        return self.model

    def get_directory(self):
        return self.directory


    def classification_report(self, X, y, plot=True, save=True, directory=None):
        if directory is None:
            directory = self.directory
        std = StandardScaler()
        std.fit(X=X)
        X = std.transform(X=X)
        self.model.load_weights(os.path.join(self.directory, 'model_weights.h5'))
        # self.model.load_weights('neuralnetwork_classifier/saved_models/lstm/model_weights.h5')

        pred = self.model.predict(X)
        y_pred = [np.round(x) for x in pred]
        report = classification_report(y_pred=y_pred, y_true=y)
        # report = classification_report(y_pred=y_pred, y_true=y, target_names=['DGA', 'Legit']) 源代码


        if save:
            self.logger.info("\n%s" % report)
            if plot:
                # plot_classification_report(classification_report=report,
                #                            directory=directory)

                self.plot_classification_report( with_avg_total=True)
        else:
            print(report)


    def fit(self, X, y, stdscaler=True, validation_data=None, validation_split=None, batch_size=5, epochs=100,
            verbose=2, early=True):
        #定义文件夹tensorboard 和本次训练出来的模型权重名字
        dirtemp = os.path.join(self.directory, "tensorboard")
        dirwe = os.path.join(self.directory, 'model_weights.h5')
        #TensorBoard 是可视化工具，到时可看到训练的过程和模型的数据
        #用法：https://blog.csdn.net/zhangpeterx/article/details/90762586
        callbacks = [
            TensorBoard(log_dir=dirtemp,
                        write_graph=False,
                        write_images=False,
                        histogram_freq=0),
            ModelCheckpoint(dirwe, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
        ]

        if early:
            callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=2, mode='auto'))
        #是否将数据去均值和方差归一化
        if stdscaler:
            std = StandardScaler()
            X = std.fit_transform(X=X)
        #开始训练数据
        self.model.fit(X, y, batch_size=batch_size,
                       epochs=epochs,
                       callbacks=callbacks,
                       validation_data=validation_data,
                       validation_split=validation_split,
                       verbose=verbose,
                       )
        # 保存模型本次训练的数据
        self.__save_model()

    def plot_AUC(self, X_test, y_test, save=True, directory=None):
        if directory is None:
            directory = self.directory
        std = StandardScaler()
        X_test = std.fit_transform(X_test)
        # self.model.load_weights(os.path.join(self.directory, 'model_weights.h5'))
        y_score = self.model.predict_proba(X_test)

        # y_score = [round(x) for x in y_score]
        # round_ = np.vectorize(round)
        # y_score = round_(y_score)

        fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_score[:, 0])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc, lw=1.5,
                 alpha=.8)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
                 label='Luck', alpha=.8)
        plt.xlim([0, 1.00])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        if save:
            dirplt = os.path.join(directory, 'roc_plot.png')
            plt.savefig(dirplt, format="png", bbox_inches='tight')
        else:
            plt.show()

    def __cross_val(self, X, y, save=False):
        #     t0 = datetime.now()
        #     self.logger.info("Starting cross validation at %s" % t0)
        #
        #     _cachedir = mkdtemp()
        #     _memory = joblib.Memory(cachedir=_cachedir, verbose=2)
        #     pipeline = Pipeline(
        #         [('standardize', StandardScaler()),
        #          ('mlp', KerasClassifier(build_fn=pierazzi_normalized_baseline,
        #                                  epochs=100,
        #                                  batch_size=5,
        #                                  verbose=2))],
        #         memory=_memory)
        #
        #     kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RandomState())
        #
        #     results = cross_validate(pipeline, X, y, cv=kfold, n_jobs=-1, verbose=2,
        #                              scoring=['precision', 'recall', 'f1', 'roc_auc'])
        #
        #     self.logger.info("Cross Validation Ended. Elapsed time: %s" % (datetime.now() - t0))
        #     if save:
        #         time.sleep(2)
        #         model = Model(pipeline.named_steps['mlp'].build_fn())
        #         model.get_model().summary(print_fn=self.logger.info)
        #
        #         model.save_results(results)
        #         model.save_model()
        #
        #         return model
        pass


def large_baseline():
    model = Sequential()

    model.add(Dense(18, input_dim=9, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])

    return model


def reduced_baseline():
    """
    Modello ridotto
    :return:
    """
    model = Sequential()

    model.add(Dense(9, input_dim=9, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(4, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_accuracy'])
    return model


def pierazzi_normalized_baseline(weights_path=None):
    model = Sequential()

    model.add(Dense(9, input_dim=9, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(128, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(64, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer='normal', use_bias=False))
    model.add(Activation('sigmoid'))

    if weights_path:
        model.load_weights(weights_path)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def pierazzi_baseline(weights_path=None):
    model = Sequential()

    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))

    model.add(Dense(128, kernel_initializer='normal', activation='relu'))

    model.add(Dense(64, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    if weights_path:
        model.load_weights(weights_path)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def pierazzi_baseline_NEW(weights_path=None):
    model = Sequential()

    model.add(Dense(15, input_dim=15, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    if weights_path:
        model.load_weights(weights_path)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def verysmall_baseline(weights_path=None):
    model = Sequential()

    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    if weights_path:
        model.load_weights(weights_path)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def lstm_baseline(maxlen, chars):
    from keras.layers import LSTM
    from keras.optimizers import RMSprop

    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    return model
