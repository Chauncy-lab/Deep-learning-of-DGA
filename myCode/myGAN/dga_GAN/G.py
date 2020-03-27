import threading

from keras.models import Model
from keras.layers import Input,Dense,LSTM,concatenate,InputLayer,Embedding,multiply,Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.optimizers import *
from keras.layers import Lambda
from keras.callbacks import TensorBoard
from myCode.dga_GAN.D import get_data,batch_generator,get_D
import keras.backend as K
from myCode.dga_GAN.env import *
import numpy as np
import random


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # python3
        with self.lock:
            return self.it.__next__()

    # def next(self): # python2
    #     with self.lock:
    #       return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def padding(x,max_len):
    shape=x.shape.as_list()
    return tf.pad(x, [[0, 0], [max_len - shape[1], 0], [0, 0]])
def padding_shape(input_shape,max_len):
    return (input_shape[0], max_len, chars_size)
def get_G():
    input_random = Input(batch_shape=(batch_size, g_domain_len, chars_size), name="input_random")
    input_root = Input(batch_shape=(batch_size, 4, chars_size), name="input_root")
    input_root_pad=Lambda(lambda x:padding(x,max_len=g_domain_len),
                          output_shape=lambda shape:padding_shape(shape,max_len=g_domain_len),
                          name="padding_root"
                          )(input_root)
    input_main = multiply([input_random, input_root_pad])
    #input_main = concatenate([input_random, input_root], axis=1)
    lstm_1 = LSTM(units=64, return_sequences=True, dropout=0.5, name="LSTM_1")(input_main)
    lstm_2 = LSTM(units=32, return_sequences=True, dropout=0.5, name="LSTM_2")(lstm_1)
    dense_1 = Dense(units=32, activation="relu", name="Dense_1")(lstm_2)
    dense_2 = Dense(units=chars_size, activation="softmax", name="Dense_2")(dense_1)
    # onehot=Lambda(to_one_hot,to_one_hot_shape,name="to_one_hot")(dense_2)
    concate = concatenate([dense_2, input_root], axis=1, name="concate")
    pad = Lambda(lambda x:padding(x,max_len=max_domain_size),
                 output_shape=lambda shape:padding_shape(shape,max_len=max_domain_size),
                 name="padding_output"
                 )(concate)
    print(pad.shape)
    G = Model(inputs=[input_random, input_root], outputs=pad, name="G")
    print(G.get_output_shape_at(0))
    return G
def get_DG(D,G):
    input_dg_random=Input(batch_shape=(batch_size,g_domain_len,chars_size))
    input_dg_root=Input(batch_shape=(batch_size,4,chars_size))
    output_dg=D(G(inputs=[input_dg_random,input_dg_root]))
    DG=Model(inputs=[input_dg_random,input_dg_root],outputs=[output_dg])
    DG.compile(optimizer=Adam(), loss=DG_loss,metrics=["accuracy"])
    #DG_D.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=["accuracy"])
    return DG
def DG_loss(y_true,y_pre):
    # return K.mean(y_pre[:][1]+y_pre[:][2]-y_pre[:][0])
    y_pre_0=tf.slice(y_pre,begin=[0,0],size=[batch_size,1])
    y_pre_1=tf.slice(y_pre,begin=[0,1],size=[batch_size,1])
    y_pre_2 = tf.slice(y_pre, begin=[0, 2], size=[batch_size, 1])
    return K.mean(y_pre_1+y_pre_2-y_pre_0)

@threadsafe_generator
def random_generator(batch_size):
    y_input = np.array([[0.0,0.0,1.0]]*batch_size)
    while True:
        root_batch = []
        for i in range(batch_size):
            root = random.choice(roots)
            root_one_hot = to_onehot(root)
            root_pad = pad_sequences([root_one_hot], maxlen=4,padding="post")[0]
            root_batch.append(root_pad)
        root_batch=np.array(root_batch)
        random_batch = np.random.normal(size=(batch_size, g_domain_len, chars_size))
        yield ([random_batch,root_batch],y_input)
def to_s(x):
    x_shape=x.shape
    m=np.max(x,axis=1,keepdims=True)
    print(m)
    condition=np.equal(m,x)
    x_one=np.where(condition,np.ones(shape=x_shape),np.zeros(shape=x_shape)).tolist()[-12:]
    s=""
    for l in x_one:
        try:
            index=l.index(1.0)
            s+=chars[index]
        except ValueError:
            pass
    if s[-1]==".":
        s=s[0:-1]
    return s


def main():
    D=load_model(D_model_path)
    # D=get_D()
    G=get_G()
    DG=get_DG(D,G)
    random_g = random_generator(batch_size=batch_size)
    data_set,data_label=get_data()
    data_generator=batch_generator(data_set,data_label,batch_size//2,epochs=10000)
    call_d = TensorBoard(log_dir="log/DG_D.log", write_grads=True)
    call_g= TensorBoard(log_dir="log/DG_G.log", write_grads=True)
    for e in range(DG_epochs):
        print("开训训练鉴别器D-[===============]-{}/{}Epochs".format(e+1, DG_epochs))
        D.trainable = True
        G.trainable = False
        for s in range(D_steps):
            r_batch,r_label=next(random_g)
            r_g=G.predict_on_batch(r_batch)
            da_batch,da_label=next(data_generator)
            d_fit_batch=r_g.tolist()[0:batch_size//2]
            d_fit_label=r_label.tolist()[0:batch_size//2]
            d_fit_batch.extend(da_batch)
            d_fit_label.extend(da_label)
            indexs=list(range(batch_size))
            random.shuffle(indexs)
            d_fit_b=[]
            d_fit_l=[]
            for index in indexs:
                d_fit_b.append(d_fit_batch[index])
                d_fit_l.append(d_fit_label[index])
            #D.fit(np.array(d_fit_b),np.array(d_fit_l),batch_size=batch_size,callbacks=[call_d])
            D.fit(np.array(d_fit_b), np.array(d_fit_l), batch_size=batch_size)
        print("开训训练生成器G-[===============]-{}/{}Epochs".format(e + 1, DG_epochs))
        D.trainable = False
        G.trainable = True
        #DG.fit_generator(random_g, steps_per_epoch=G_steps,callbacks=[call_g])
        DG.fit_generator(random_g, steps_per_epoch=G_steps)
        if e%10==0:
            a = random_g.__next__()[0]
            one_pre = G.predict_on_batch(a)
            for n in range(10):
                print(to_s(one_pre[n]))
    D.save("model/DG_D_model")
    G.save("model/DG_G_model")
    DG.save("model/DG_model")
        # DG_D.fit_generator(random_g,steps_per_epoch=(D_steps//3)*2)
        # D.fit_generator(data_generator,steps_per_epoch=D_steps//3)


if __name__=="__main__":
    main()