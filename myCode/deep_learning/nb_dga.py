import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn import metrics
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

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

def get_aeiou(domain):#统计每个域名元音字母个数
    count = len(re.findall(r'[aeiou]', domain.lower())) #将所有字母转为小写，并且检测是否有元音字母，返回匹配该该域名的匹配的个数，比如goole.com返回的是4
    #count = (0.0 + count) / len(domain)
    return count

def get_uniq_char_num(domain):#统计每个域名不重复字符个数
    count=len(set(domain))
    #count=(0.0+count)/len(domain)
    return count

def get_uniq_num_num(domain):#统计每个域名数字个数
    count = len(re.findall(r'[1234567890]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count


def get_feature():
    from sklearn import preprocessing
    alexa=load_alexa()
    dga=load_dga()
    v=alexa+dga      #正常域名和dga域名数组合并
    y=[0]*len(alexa)+[1]*len(dga)    #将数据打标，正常为了0，dga为1，标签集合
    x=[]
    for vv in v:
        vvv=[get_aeiou(vv),get_uniq_char_num(vv),get_uniq_num_num(vv),len(vv)] #手工提取三个特征向量值来标记一个域名
        x.append(vvv) #样本集合

    x=preprocessing.scale(x) #可以直接将给定数据进行标准化
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)#分别返回x（样本集合）和y（标签集合）的训练集和测试集
    return x_train, x_test, y_train, y_test

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
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    x = metrics.auc(fpr, tpr)
    print(x)

    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve of %s (AUC = %.4f)" % ('lightgbm', x))
    plt.legend(loc="lower right")
    plt.show()


    # y_pred=gnb.predict(x_test)   #将测试样本进行预测
    # print(classification_report(y_test, y_pred))  #与原本真实的标签测试集合进行对比，产生报告
    # print (metrics.confusion_matrix(y_test, y_pred)) #与原本真实的标签测试集合进行对比，产生混淆矩阵


if __name__ == "__main__":
    print("Hello dga")
    print("text feature & nb")
    x_train, x_test, y_train, y_test = get_feature()
    do_nb(x_train, x_test, y_train, y_test)

    # print( "2-gram & nb")
    # x_train, x_test, y_train, y_test = get_feature_2gram()
    # do_nb(x_train, x_test, y_train, y_test)