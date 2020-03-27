# -*- coding:utf-8 -*-

import sys
import urllib

import sklearn
# import urlparse
import re
from hmmlearn import hmm
import numpy as np
from sklearn.externals import joblib
# import HTMLParser
import nltk
import csv
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB

#处理域名的最小长度
MIN_LEN=10

#状态个数
N=8
#最大似然概率阈值
T=-50

#模型文件名
FILE_MODEL="9-2.m"

def load_alexa(filename):
    domain_list=[]
    csv_reader = csv.reader(open(filename)) #python csv模块读写文件，返回一个list
    for row in csv_reader:
        domain=row[1] #选取第二个，row[0]是前面的序号
        if len(domain)>= MIN_LEN: #若域名长度>=10
            domain_list.append(domain) #放进domain_list里面
    return domain_list

def domain2ver(domain):
    ver=[]
    for i in range(0,len(domain)):
        ver.append([ord(domain[i])])
    return ver


def load_dga(filename):
    domain_list=[]
    #xsxqeadsbgvpdke.co.uk,Domain used by Cryptolocker - Flashback DGA for 13 Apr 2017,2017-04-13,
    # http://osint.bambenekconsulting.com/manual/cl.txt
    with open(filename) as f:
        for line in f:  #读取text文件，每一行每一行读取
            domain=line.split(",")[0] # 以逗号为分割,选取第一个
            if len(domain)>= MIN_LEN:# 如果域名>=10
                domain_list.append(domain) #放进domain_list中
    return  domain_list

def test_dga(remodel,filename):
    x=[]
    y=[]
    dga_cryptolocke_list = load_dga(filename)
    for domain in dga_cryptolocke_list:
        domain_ver=domain2ver(domain)
        np_ver = np.array(domain_ver)
        pro = remodel.score(np_ver)
        #print  "SCORE:(%d) DOMAIN:(%s) " % (pro, domain)
        x.append(len(domain))
        y.append(pro)
    return x,y

def test_alexa(remodel,filename):
    x=[]
    y=[]
    alexa_list = load_alexa(filename)
    for domain in alexa_list:
        domain_ver=domain2ver(domain)
        np_ver = np.array(domain_ver)
        pro = remodel.score(np_ver)
        #print  "SCORE:(%d) DOMAIN:(%s) " % (pro, domain)
        x.append(len(domain))
        y.append(pro)
    return x, y


def nb_dga():
    x1_domain_list = load_alexa("../../data/top-1000.csv")
    x2_domain_list = load_dga("../../data/dga-cryptolocke-1000.txt")
    x3_domain_list = load_dga("../../data/dga-post-tovar-goz-1000.txt")

    x_domain_list=np.concatenate((x1_domain_list, x2_domain_list,x3_domain_list)) #concatenate拼接数组，默认axis=0: 合并行

    y1=[0]*len(x1_domain_list)
    y2=[1]*len(x2_domain_list)
    y3=[2]*len(x3_domain_list)

    y=np.concatenate((y1, y2,y3))

    #测试代码部分（可删）

    # data = ["为了祖国，为了胜利，向我开炮！向我开炮！",
    #         "记者：你怎么会说出那番话",
    #         "我只是觉得，对准我自己打"]
    #
    # cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",token_pattern=r"\w", min_df=1)
    # x = cv.fit_transform(data)
    # print(cv.get_feature_names())
    # print(x)
    # print(x.toarray())
    # print(cv.vocabulary_)

    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore", token_pattern=r"\w", min_df=1)
    x= cv.fit_transform(x_domain_list).toarray()

    clf = GaussianNB()
    print(sklearn.model_selection.cross_val_score(clf, x, y, n_jobs=-1, cv=3))

if __name__ == '__main__':
    nb_dga()