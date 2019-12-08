import tflearn
import pandas as pd
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics

dga_file="../../data/dga/dga.txt"
alexa_file="../../data/dga/top-1m.csv"

def load_alexa():
    x=[]
    data = pd.read_csv(alexa_file, sep=",",header=None)
    x=[i[1] for i in data.values]
    return x

def load_dga():
    x=[]
    data = pd.read_csv(dga_file, sep="\t", header=None,
                      skiprows=18)
    x=[i[1] for i in data.values]
    return x

def get_feature_charseq():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    t=[]
    for i in x:                        #字符转ASCII值，把所有域名转换为数字
        v=[]
        for j in range(0,len(i)):
            v.append(ord(i[j]))
        t.append(v)

    x=t
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)

    return x_train, x_test, y_train, y_test

def do_rnn(trainX, testX, trainY, testY):
    max_document_length=64
    y_test=testY
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.) #二值化，因为域名长度不一，统一转成长度（64）相同的数值运行，不够的话补0
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2) #将整型标签转为onehot，返回的是01矩阵
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, max_document_length])
    # input_dim是看分语料中取了多少个单词，一般取top多少进行output_dim 就是得到 embedding 向量的维度
    net = tflearn.embedding(net, input_dim=10240000, output_dim=64)
    net = tflearn.lstm(net, 64, dropout=0.1)#net 传进来的张量，64是此层的单位数
    net = tflearn.fully_connected(net, 2, activation='softmax')#全连接层，2层
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
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print (metrics.confusion_matrix(y_test, y_predict))



if __name__ == "__main__":
    print ("Hello dga")
    print("charseq & rnn")
    x_train, x_test, y_train, y_test = get_feature_charseq()
    do_rnn(x_train, x_test, y_train, y_test)