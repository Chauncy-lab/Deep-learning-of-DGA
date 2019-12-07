import re
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import numpy as np

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

def get_aeiou(domain):
    count = len(re.findall(r'[aeiou]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_uniq_char_num(domain):
    count=len(set(domain))
    #count=(0.0+count)/len(domain)
    return count

def get_uniq_num_num(domain):
    count = len(re.findall(r'[1234567890]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_feature():
    from sklearn import preprocessing
    alexa=load_alexa()
    dga=load_dga()
    v=alexa+dga
    y=[0]*len(alexa)+[1]*len(dga)
    x=[]

    for vv in v:
        vvv=[get_aeiou(vv),get_uniq_char_num(vv),get_uniq_num_num(vv),len(vv)]
        x.append(vvv)

    x=preprocessing.scale(x)
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)
    return x_train, x_test, y_train, y_test

def get_feature_234gram():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    CV = CountVectorizer(
                                    ngram_range=(2, 4),
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

def do_mlp(x_train, x_test, y_train, y_test):

    global max_features
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs', #quasi-Newton方法的优化器
                        alpha=1e-5,  #正则化项参数
                        hidden_layer_sizes = (5, 2),#两层隐藏层，第一层隐藏层有5个神经元，第二层也有2个神经元
                        random_state = 1) #int 或RandomState，可选，默认None，随机数生成器的状态或种子
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))


# def test():
#     alexa = load_alexa()
#     dga = load_dga()
#     X = alexa + dga
#     y = [0] * len(alexa) + [1] * len(dga)
#     best_iter = 0
#
#     ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2), dtype=np.int32)
#     count_vec = ngram_vectorizer.fit_transform(X) # 数据拟合和标准化
#     max_features = count_vec.shape[1]    # 计算输入维度
#     # 划分训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(count_vec, y,  test_size=0.2)
#     X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
#
#     #创建一个Sequential模型
#     model = Sequential() # 添加一个全连接层，激活函数使用sigmoid，输出维度128
#     model.add(Dense(128, input_dim=max_features, activation='sigmoid'))  # 添加一个Dropout层，用于防止过拟合
#     model.add(Dropout(0.5))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid')) #编译模型，损失函数采用对数损失函数，优化器选用adam
#     model.compile(loss='binary_crossentropy',optimizer='adam')
#
#     for ep in range(50):
#         model.fit(X_train, y_train, batch_size=32, nb_epoch=1)
#         t_probs = model.predict_proba(X_holdout)
#         t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs) # 计算AUC值
#         print("t_auc:"+str(t_auc))
#         if t_auc > 32:
#             best_auc = t_auc
#             best_iter = ep
#             # 打印分类模型指标
#             probs = model.predict_proba(X_test)
#             print(classification_report(y_test, probs > .5))
#             print('mlp: auc = %f ' % metrics.confusion_matrix(y_test, probs > .5))
#         else:
#             if (ep - best_iter) > 2:
#                 break


if __name__ == "__main__":
    print ("Hello dga")
    # print( "text feature & mlp")
    # x_train, x_test, y_train, y_test = get_feature()
    # do_mlp(x_train, x_test, y_train, y_test)
    # print ("234-gram & mlp")
    # x_train, x_test, y_train, y_test = get_feature_234gram()
    # do_mlp(x_train, x_test, y_train, y_test)
    print("2-gram & mlp")
    x_train, x_test, y_train, y_test = get_feature_2gram()
    do_mlp(x_train, x_test, y_train, y_test)