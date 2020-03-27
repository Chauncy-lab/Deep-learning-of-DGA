import time

import pandas as pd
from keras import optimizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from myCode.reinforcement_learning_GAN.dga_GAN.dga_gan import __build_dataset

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

# 试图加载没有带后缀.com的正常域名和GAN生成的DGA 进入模型比较
# def load_alexa():
#     x=[]
#     # data = pd.read_csv("../../data/dga/legitdomains.txt", sep=" ",header=None)
#     # x=[i[1] for i in data.values]
#     # 加载正常域名（测试gan）
#     data_dict = __build_dataset()
#     return data_dict['legit_domain'].tolist()
#
#
# def load_dga():
#     x=[]
#     data = pd.read_csv(dga_file, sep="\t", header=None,
#                       skiprows=18)
#     x=[i[1] for i in data.values]
#
#     a = []
#     b = []
#     #测试gan
#     for j in range(0,len(x)):
#       a =  x[j].split('.')
#       b.append(a[0])
#     return b




def test(learn_late,dropout_rate):
    alexa = load_alexa()
    dga = load_dga()
    X = alexa + dga
    y = [0] * len(alexa) + [1] * len(dga)
    best_iter = -1
    best_auc = 0.0
    best_acc = 0.0

    ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2), dtype=np.int32)
    count_vec = ngram_vectorizer.fit_transform(X) # 数据拟合和标准化
    max_features = count_vec.shape[1]    # 计算输入维度
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(count_vec, y,  test_size=0.2)
    X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)

    #创建一个Sequential模型
    model = Sequential() # 添加一个全连接层，激活函数使用sigmoid，输出维度128
    model.add(Dense(128, input_dim=max_features, activation='sigmoid'))  # 添加一个Dropout层，用于防止过拟合
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    adam =optimizers.adam(lr=learn_late)
    # sdg = optimizers.SGD(lr=learn_late)

    #编译模型，损失函数采用对数损失函数，优化器选用adam
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    #开始训练时间
    start = time.clock()
    for ep in range(50):
        model.fit(X_train, y_train, batch_size=128, nb_epoch=1)
        t_probs = model.predict_proba(X_holdout)
        # 另一种打印矩阵报告的方式，分类模型指标
        # t_probs = np.rint(t_probs)
        # print(classification_report(y_holdout, t_probs))
        # print(metrics.confusion_matrix(y_holdout, t_probs))
        t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs) # 计算AUC值
        t_acc = metrics.accuracy_score(y_holdout, t_probs > 0.5)# 计算准确率
        # print("t_auc:"+str(t_auc))
        if t_auc > best_auc:
            best_auc = t_auc
            best_acc = t_acc
            best_iter = ep

        else:
            if (ep - best_iter) > 2:
                # 打印分类模型指标
                probs = model.predict_proba(X_test)
                print(accuracy_score(y_test, probs > .5))
                print(classification_report(y_test, probs > .5))
                print(metrics.confusion_matrix(y_test, probs > .5))
                fpr, tpr, threshold = roc_curve(y_holdout, t_probs)
                x = metrics.auc(fpr, tpr)
                print(x)
                roc_auc = auc(fpr, tpr)


                # plt.figure()
                # lw = 2
                # plt.figure(figsize=(10, 10))
                # plt.plot(fpr, tpr, color='darkorange', lw=lw,
                #          label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
                # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                # plt.xlim([0.0, 1.0])
                # plt.ylim([0.0, 1.05])
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')
                # plt.title("mlp ROC curve of %s (AUC = %.4f)" % ('lightgbm', x))
                # plt.legend(loc="lower right")
                # plt.show()
                break
    #结束时间
    end = time.clock()
    #50轮的训练耗时
    train_time = end-start
    print(train_time)
    #输出实验数据，字典形式，每个惩罚系数对应的最优auc
    d = {}
    d[dropout_rate]=best_auc
    #每个惩罚系数对应的最优acc
    g={}
    g[dropout_rate]=best_acc
    # 每个惩罚系数对应的最优auc的迭代
    f={}
    f[dropout_rate]=best_iter
    # 每个惩罚系数对应的最优auc所需要的时间
    t={}
    t[dropout_rate] = train_time

    return d,g,f,t

def run_test():
    lr = [0.001, 0.01, 0.1, 0.2, 0.3]
    dr =[0.1,0.3,0.5,0.7,0.9]
    auc_list=[]
    acc_list=[]
    iter_list=[]
    time_list=[]
    for i in lr:
        for  j in dr:
           new_best_auc,new_best_acc,new_best_iter,new_time = test(i,j)
           auc_list.append(new_best_auc)
           acc_list.append(new_best_acc)
           iter_list.append(new_best_iter)
           time_list.append(new_time)
    print(auc_list)
    print(acc_list)
    print(iter_list)
    print(time_list)

if __name__ == "__main__":
    run_test()
    # a = [{0.1: 0.9971868670450852}, {0.3: 0.99795}, {0.5: 0.99718125}, {0.7: 0.9964920852673845}, {0.9: 0.6536400122721666}, {0.1: 0.9968943904928277}, {0.3: 0.9977182366032358}, {0.5: 0.9972357723577236}, {0.7: 0.9966364699189126}, {0.9: 0.996862009689014}, {0.1: 0.9934172688854992}, {0.3: 0.9982801751094434}, {0.5: 0.9922742968671805}, {0.7: 0.997793694842371}, {0.9: 0.5}, {0.1: 0.9947800755169913}, {0.3: 0.9965624140603515}, {0.5: 0.9936724301871683}, {0.7: 0.98638125}, {0.9: 0.5}, {0.1: 0.9963554299717436}, {0.3: 0.9952644880392353}, {0.5: 0.9963211100596018}, {0.7: 0.9290752620873102}, {0.9: 0.5}]
    # print(len(a))
    # q= [0.8320676691729324, 0.8437552829798824, 0.7583790142925084, 0.47900149787852764, 0.53830625, 0.9655135737122336,
    #  0.9322008390240231, 0.8096577414435361, 0.6395583841731315, 0.6956157462985194, 0.997630716911305,
    #  0.9930305530553055, 0.9911654135338346, 0.7736926461345066, 0.5256933473758119, 0.9968559355935593,
    #  0.9960124750779692, 0.9956248906222656, 0.7802921816902824, 0.5, 0.9975484677923703, 0.9970669168230144,
    #  0.9974433505235194, 0.9926109453109847, 0.5]
    # print(len(q))
