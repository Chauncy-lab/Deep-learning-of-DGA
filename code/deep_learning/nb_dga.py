import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import metrics
import re
from sklearn.model_selection import train_test_split

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


def do_nb(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    print("Hello dga")
    print("text feature & nb")
    x_train, x_test, y_train, y_test = get_feature()
    do_nb(x_train, x_test, y_train, y_test)