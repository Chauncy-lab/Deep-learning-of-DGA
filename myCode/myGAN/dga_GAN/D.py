from keras.models import Model
from keras.layers import Input,Dense,LSTM,Flatten
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,precision_score
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend as K
import numpy as np
from myCode.dga_GAN.env import *


def get_D():
    input=Input(shape=(max_domain_size,chars_size),name="D_Input")
    l_1=LSTM(units=64,activation="tanh",dropout=0.5,return_sequences=True,name="D_LSTM_1")(input)
    l_2=LSTM(units=32,activation="relu",return_sequences=True,dropout=0.5,name="D_LSTM_2")(l_1)
    flatten=Flatten()(l_2)
    d_1=Dense(units=16,activation="relu",name="Dense_1")(flatten)
    output=Dense(units=3,name="Output",activation="softmax")(d_1)
    D=Model(inputs=input,outputs=output)
    D.fit_generator
    D.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=["accuracy"])
    return D
def  batch_generator(X,Y,batch_size,epochs=1):
    x_batch=[]
    y_batch=[]
    n=0
    for e in range(epochs):
        X_onehot=map(to_onehot, X)
        X_pad = map(lambda d: pad_sequences([d], maxlen=max_domain_size, truncating="pre")[0], X_onehot)
        for index,label in enumerate(Y):
            x_batch.append(X_pad.__next__())
            y_batch.append(label)
            n+=1
            if n==batch_size:
                yield (np.array(x_batch),np.array(y_batch))
                x_batch=[]
                y_batch=[]
                n=0


def train(X_train,Y_train):
    D = get_D()
    call_1 = TensorBoard(log_dir="log/D.log", write_grads=True)
    train_generator = batch_generator(X_train, Y_train, batch_size=batch_size, epochs=D_epochs)
    D.fit_generator(train_generator, steps_per_epoch=len(Y_train)//batch_size, epochs=D_epochs, callbacks=[call_1])
    D.save(D_model_path)
def test(X_test,Y_test):
    G=load_model(D_model_path)
    test_generator=batch_generator(X_test,Y_test,batch_size=batch_size)
    Y_pre=[]
    Y_true=[]
    steps=0
    batch_num=len(Y_test)//batch_size
    while True:
        try:
            x_batch,y_batch=test_generator.__next__()
        except StopIteration:
            break
        Y_true.extend(y_batch)
        y_pre=G.predict_on_batch(x_batch)
        Y_pre.extend(y_pre.round())
        steps+=1
        print("%d/%d batch" % (steps, batch_num))
    Y_true=np.array(Y_true)
    Y_pre=np.array(Y_pre)
    precision=precision_score(Y_true[:,1:2],Y_pre[:,1:2])
    recall=recall_score(Y_true[:,1:2],Y_pre[:,1:2])
    print("P:",precision)
    print("R:",recall)
def get_data():
    dga_list = []
    with open("data/dga.txt") as f:
        for l in f.readlines():
            domain = l.split("\t")[1].strip()
            dga_list.append(domain)
    top_list = []
    with open("data/top-1m.csv") as f:
        for l in f.readlines():
            domain = l.split(",")[1].strip()
            top_list.append(domain)
    dga_label = [[0.0, 1.0, 0.0]] * len(dga_list)
    top_label = [[1.0, 0.0, 0.0]] * len(top_list)
    data_set = dga_list + top_list
    data_label = dga_label + top_label
    return data_set,data_label
def main():
    data_set,data_label=get_data()
    X_train,X_test,Y_train,Y_test=train_test_split(data_set,data_label,test_size=0.3)
    train(X_train,Y_train)
    test(X_test,Y_test)
if __name__=="__main__":
    K.set_session(tf.Session())
    main()