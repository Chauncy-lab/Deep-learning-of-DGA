import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
import sklearn

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

def get_feature_2gram():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    CV = CountVectorizer(ngram_range=(2, 2),token_pattern=r'\w',decode_error='ignore',strip_accents='ascii',
                         max_features=max_features,
                         stop_words='english',
                         max_df=1.0,
                         min_df=1)
    x = CV.fit_transform(x)
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4)

    return x_train.toarray(), x_test.toarray(), y_train, y_test

def do_xgboost(x_train, x_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))

    # print(metrics.accuracy_score(y_test, y_pred))
    # t_probs = xgb_model.predict_proba(x_test)[:, 1]
    # t_auc = sklearn.metrics.roc_auc_score(y_test, t_probs)
    # print('XGBClassifier: auc = %f ' % (t_auc))

if __name__ == "__main__":
    print("text feature & xgboost")
    x_train, x_test, y_train, y_test = get_feature()
    do_xgboost(x_train, x_test, y_train, y_test)

    print("2-gram & XGBoost")
    x_train, x_test, y_train, y_test = get_feature_2gram()
    do_xgboost(x_train, x_test, y_train, y_test)
