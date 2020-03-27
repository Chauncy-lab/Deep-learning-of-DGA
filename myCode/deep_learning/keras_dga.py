import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve,auc
import sklearn
import numpy as np
import matplotlib.pyplot as plt


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


def test():
    alexa = load_alexa()
    dga = load_dga()
    X = alexa + dga
    y = [0] * len(alexa) + [1] * len(dga)
    best_iter = -1
    best_auc = 0.0

    ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2), dtype=np.int32)
    count_vec = ngram_vectorizer.fit_transform(X) # 数据拟合和标准化
    max_features = count_vec.shape[1]    # 计算输入维度
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(count_vec, y,  test_size=0.2)
    X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)

    #创建一个Sequential模型
    model = Sequential() # 添加一个全连接层，激活函数使用sigmoid，输出维度128
    model.add(Dense(128, input_dim=max_features, activation='sigmoid'))  # 添加一个Dropout层，用于防止过拟合
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid')) #编译模型，损失函数采用对数损失函数，优化器选用adam
    model.compile(loss='binary_crossentropy',optimizer='adam')

    for ep in range(50):
        model.fit(X_train, y_train, batch_size=128, nb_epoch=1)
        t_probs = model.predict_proba(X_holdout)
        t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs) # 计算AUC值
        print("t_auc:"+str(t_auc))
        if t_auc > best_auc:
            best_auc = t_auc
            best_iter = ep
            # 打印分类模型指标
            probs = model.predict_proba(X_test)
            print(classification_report(y_test, probs > .5))
            print(metrics.confusion_matrix(y_test,probs > .5))
        else:
            if (ep - best_iter) > 2:

                fpr, tpr, threshold = roc_curve(y_holdout, t_probs)
                x = metrics.auc(fpr, tpr)
                print(x)
                roc_auc = auc(fpr, tpr)
                plt.figure()
                lw = 2
                plt.figure(figsize=(10, 10))
                plt.plot(fpr, tpr, color='darkorange', lw=lw,
                         label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title("mlp ROC curve of %s (AUC = %.4f)" % ('lightgbm', x))
                plt.legend(loc="lower right")
                plt.show()

                break




if __name__ == "__main__":
    test()