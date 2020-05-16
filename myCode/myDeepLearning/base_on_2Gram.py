import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn import metrics
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from tflearn.data_utils import to_categorical, pad_sequences
import tflearn

dga_file="../../data/dga/dga.txt"
alexa_file="../../data/dga/top-1m.csv"

#加载alexa文件中的数据
def load_alexa():
    x=[]
    data = pd.read_csv(alexa_file, sep=",",header=None)
    x=[i[1] for i in data.values]
    return x

#加载dga数据
def load_dga():
    x=[]
    data = pd.read_csv(dga_file, sep="\t", header=None,skiprows=18) #跳过前18行注释
    x=[i[1] for i in data.values]
    return x



def get_feature_2gram():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    CV = CountVectorizer(
                                    ngram_range=(2, 2),
                                    token_pattern=r'\w',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    max_features=max_features,
                                    stop_words='english',
                                    max_df=1.0,
                                    min_df=1)
    x = CV.fit_transform(x)
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)

    return x_train.toarray(), x_test.toarray(), y_train, y_test



def do_nb(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train,y_train) #使用朴素贝叶斯训练
    y_pred = gnb.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    return y_pred, y_test

def do_xgboost(x_train, x_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    return y_pred,y_test

def do_mlp(x_train, x_test, y_train, y_test):

    global max_features
    # Building deep neural network
    clf = MLPClassifier(solver='adam', #quasi-Newton方法的优化器
                        alpha=1e-5,  #正则化项参数
                        hidden_layer_sizes = (5, 2),#两层隐藏层，第一层隐藏层有5个神经元，第二层也有2个神经元
                        random_state = 1) #int 或RandomState，可选，默认None，随机数生成器的状态或种子
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    return y_pred,y_test

def do_rnn(trainX, testX, trainY, testY):
    max_document_length=64
    y_test=testY
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, max_document_length])
    net = tflearn.embedding(net, input_dim=10240000, output_dim=64)
    net = tflearn.lstm(net, 64, dropout=0.1)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0,tensorboard_dir="dga_log")
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=10,run_id="dga",n_epoch=1)

    y_predict_list = model.predict(testX)
    #print y_predict_list

    y_predict = []
    for i in y_predict_list:
        print  (i[0])
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print(metrics.confusion_matrix(y_test, y_predict))
    return y_predict, y_test



def sum_plt(nb_predict,nb_test,xgb_predict,xgb_test,mlp_predict,mlp_test, rnn_predict, rnn_test):
    # nb的预测数组
    nb_fpr, nb_tpr, mlp_threshold = roc_curve(nb_test, nb_predict)
    #xgboost的预测数组
    xgb_fpr, xgb_tpr, xgb_threshold = roc_curve(xgb_test, xgb_predict)
    # mlp的预测数组
    mlp_fpr, mlp_tpr, mlp_threshold = roc_curve(mlp_test, mlp_predict)
    # rnn的预测数组
    rnn_fpr, rnn_tpr, rnn_threshold = roc_curve(rnn_test, rnn_predict)

    xgb = metrics.auc(xgb_fpr, xgb_tpr)
    mlp = metrics.auc(mlp_fpr, mlp_tpr)
    nb = metrics.auc(nb_fpr, nb_tpr)
    rnn = metrics.auc(rnn_fpr, rnn_tpr)

    print("nb auc= (%0.2f)" % nb)
    print(classification_report(nb_test, nb_predict))
    print(metrics.confusion_matrix(nb_test, nb_predict))
    print("xgb auc= (%0.2f)" % xgb)
    print(classification_report(xgb_test, xgb_predict))
    print(metrics.confusion_matrix(xgb_test, xgb_predict))
    print("mlp auc= (%0.2f)" % mlp)
    print(classification_report(mlp_test, mlp_predict))
    print(metrics.confusion_matrix(mlp_test, mlp_predict))
    print("rnn auc= (%0.2f)" % rnn)
    print(classification_report(rnn_test, rnn_predict))
    print(metrics.confusion_matrix(rnn_test, rnn_predict))

    xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
    mlp_roc_auc = auc(mlp_fpr, mlp_tpr)
    nb_roc_auc = auc(nb_fpr, nb_tpr)
    rnn_roc_auc = auc(rnn_fpr, rnn_tpr)

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(xgb_fpr, xgb_tpr, marker ='s',ms=10,color='purple', lw=lw, label='xgb_ROC curve (area = %0.2f)' % xgb_roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(mlp_fpr, mlp_tpr, marker ='*',ms=10,color='darkorange', lw=lw, label='mlp_ROC curve (area = %0.2f)' % mlp_roc_auc)
    plt.plot(nb_fpr, nb_tpr, marker='v', ms=10, color='green', lw=lw,label='nb_ROC curve (area = %0.2f)' % nb_roc_auc)
    plt.plot(rnn_fpr, rnn_tpr, marker='p', ms=10, color='yellow', lw=lw, label='rnn_ROC curve (area = %0.2f)' % rnn_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("2-gram sum comparison")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    print("compare dga")


    print( "2-gram & nb")
    x_train, x_test, y_train, y_test = get_feature_2gram()
    nb_predict,nb_test = do_nb(x_train, x_test, y_train, y_test)


    print("2-gram & XGBoost")
    x_train, x_test, y_train, y_test = get_feature_2gram()
    xgb_predict,xgb_test = do_xgboost(x_train, x_test, y_train, y_test)


    print("2-gram & mlp")
    x_train, x_test, y_train, y_test = get_feature_2gram()
    mlp_predict,mlp_test=do_mlp(x_train, x_test, y_train, y_test)


    print("2-gram & rnn")
    x_train, x_test, y_train, y_test = get_feature_2gram()
    rnn_predict, rnn_test=do_rnn(x_train, x_test, y_train, y_test)

    sum_plt(nb_predict,nb_test,xgb_predict,xgb_test,mlp_predict,mlp_test, rnn_predict, rnn_test)