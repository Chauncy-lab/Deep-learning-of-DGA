import argparse
import logging
import os
import string
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras.optimizers import RMSprop, adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Conv1D, Dropout, concatenate, LSTM, RepeatVector, Dense, TimeDistributed, \
    LeakyReLU, BatchNormalization, AveragePooling1D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
import matplotlib as mpl


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CUDA_VISIBLE_DEVICES = 0


# K.set_learning_phase(0)
# print("set learning phase to %s" % K.learning_phase())


def generator_model(summary=True, print_fn=None):
    """
    Generator model:
    # In: (batch_size, 1),
    # Out: (batch_size, timesteps, word_index)
    :param summary: set to True to have a summary printed to output and a plot to file at "images/discriminator.png"
    :return: generator model
    """
    dropout_value = 0.4
    cnn_filters = [20, 10]
    cnn_kernels = [2, 3]
    cnn_strides = [1, 1]
    dec_convs = []
    leaky_relu_alpha = 0.2
    latent_vector = 20
    timesteps = 15
    word_index = 38

    dec_inputs = Input(shape=(latent_vector,),
                       name="Generator_Input")
    # decoded = Dense(word_index)(dec_inputs)
    # decoded = BatchNormalization(momentum=0.9)(decoded)
    # decoded = LeakyReLU(leaky_relu_alpha)(decoded)
    # decoded = Dropout(dropout_value)(decoded)
    decoded = RepeatVector(timesteps, name="gen_repeate_vec")(dec_inputs)
    decoded = LSTM(word_index, return_sequences=True, name="gen_LSTM")(decoded)
    decoded = Dropout(dropout_value)(decoded)
    for i in range(2):
        conv = Conv1D(cnn_filters[i],
                      cnn_kernels[i],
                      padding='same',
                      strides=cnn_strides[i],
                      name='gen_conv%s' % i)(decoded)
        # conv = BatchNormalization(momentum=0.9)(conv)
        conv = LeakyReLU(alpha=leaky_relu_alpha)(conv)
        conv = Dropout(dropout_value, name="gen_dropout%s" % i)(conv)
        dec_convs.append(conv)

    decoded = concatenate(dec_convs)
    decoded = TimeDistributed(Dense(word_index, activation='softmax'), name='decoder_end')(
        decoded)  # output_shape = (samples, maxlen, max_features )

    G = Model(inputs=dec_inputs, outputs=decoded, name='Generator')
    if summary:
        if print_fn:
            G.summary(print_fn=print_fn)
        G.summary()
    return G


def discriminator_model(summary=True, print_fn=None):
    """
    Discriminator model takes a 3D tensor of size (batch_size, timesteps, word_index), outputs a domain embedding tensor of size (batch_size, lstm_vec_dim).
    :param summary: set to True to have a summary printed to output and a plot to file at "images/discriminator.png"
    :return: Discriminator model
    """
    dropout_value = 0.5
    cnn_filters = [20, 10]
    cnn_kernels = [2, 3]
    cnn_strides = [1, 1]
    enc_convs = []
    embedding_vec = 20  # lunghezza embedding layer
    leaky_relu_alpha = 0.2
    timesteps = 15
    word_index = 38
    latent_vector = 20

    discr_inputs = Input(shape=(timesteps, word_index),
                         name="Discriminator_Input")
    # embedding layer. expected output ( batch_size, timesteps, embedding_vec)
    # manual_embedding = Dense(embedding_vec, activation='linear', name="manual_embedding")
    # discr = TimeDistributed(manual_embedding, name='embedded', trainable=False)(discr_inputs)
    # discr = Embedding(word_index, embedding_vec, input_length=timesteps, name="discr_embedd")(
    #     discr_inputs)  # other embedding layer
    for i in range(2):
        conv = Conv1D(cnn_filters[i],
                      cnn_kernels[i],
                      padding='same',
                      strides=cnn_strides[i],
                      name='discr_conv%s' % i)(discr_inputs)
        conv = BatchNormalization()(conv)
        conv = LeakyReLU(alpha=leaky_relu_alpha)(conv)
        conv = Dropout(dropout_value, name='discr_dropout%s' % i)(conv)
        conv = AveragePooling1D()(conv)
        enc_convs.append(conv)

    # concatenating CNNs. expected output (batch_size, 7, 30)
    discr = concatenate(enc_convs)
    # discr = Flatten()(discr)
    discr = LSTM(latent_vector)(discr)
    # discr = Dropout(dropout_value)(discr)
    discr = Dense(1, activation='sigmoid',
                  kernel_initializer='normal'
                  )(discr)

    D = Model(inputs=discr_inputs, outputs=discr, name='Discriminator')

    if summary:
        if print_fn:
            D.summary(print_fn=print_fn)
        else:
            D.summary()
            # plot_model(D, to_file="images/discriminator.png", show_shapes=True)
    return D


def adversarial(g, d):
    """
    Adversarial Model
    :param g: Generator
    :param d: Discriminator
    :return: Adversarial model
    """
    adv_model = Sequential()
    adv_model.add(g)
    d.trainable = False
    adv_model.add(d)
    return adv_model


def train(BATCH_SIZE=32, disc=None, genr=None, original_model_name=None, weights=False):
    #定义本次训练的保存的文件名字
    directory = os.path.join("experiments", datetime.now().strftime("%Y%m%d-%H%M%S"))
    #创建保存本次模型的参数文件夹
    if not os.path.exists(directory):
        # crea la cartella
        os.makedirs(directory)
        os.makedirs(directory + "/model")
    # 创建保存本次模型的输出日志
    hdlr = logging.FileHandler(os.path.join(directory, 'output.log'))
    logger.addHandler(hdlr)
    logger.debug(directory)

    if original_model_name is not None:
        logger.debug("MORE TRAINING on the model %s" % original_model_name)

    # load dataset（加载数据集）
    latent_dim = 20
    maxlen = 15
    n_samples = 25000
    data_dict = __build_dataset(maxlen=maxlen, n_samples=int(n_samples + n_samples * 0.33))
    #训练集
    X_train = data_dict['X_train']
    #训练集的张量（形状）
    print("Training set shape %s" % (X_train.shape,))

    #如果是没引用已存在的判别器和生成器，那就创建本次模型记录本次判别器和生成器的参数设置，以图片形式输出
    # models
    if disc is None:
        disc = discriminator_model(print_fn=logger.debug)
        plot_model(disc, to_file=os.path.join(directory, "discriminator.png"), show_shapes=True)
    if genr is None:
        genr = generator_model(print_fn=logger.debug)
        plot_model(genr, to_file=os.path.join(directory, "generator.png"), show_shapes=True)
    #引用之前训练过的权重
    if weights:
        disc.load_weights1(filepath='autoencoder_experiments/20171218-101804/weights/autoencoder.h5',by_name=True)
        genr.load_weights2(filepath='autoencoder_experiments/20171218-101804/weights/autoencoder.h5',by_name=True)
    #连接为对抗模型
    gan = adversarial(genr, disc)

    #   optimizers（优化）
    discr_opt = RMSprop(
        lr=0.01,
        clipvalue=1.0,
        decay=1e-8)
    # gan_opt = RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8) #usual
    gan_opt = adam(
        lr=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        decay=1e-8,
        clipvalue=1.0)  # alternative

    #   compilation
    gan.compile(loss='binary_crossentropy', optimizer=discr_opt)
    disc.trainable = True
    disc.compile(loss='binary_crossentropy', optimizer=gan_opt)
    gan.summary(print_fn=logger.debug)

    # callbacks（可视化整个训练过程）
    tb_gan = TensorBoard(log_dir=os.path.join(directory, ".log/gan"), write_graph=True, histogram_freq=0,
                         batch_size=BATCH_SIZE)
    tb_gan.set_model(gan)
    tb_disc = TensorBoard(log_dir=os.path.join(directory, ".log/disc"), write_graph=True, histogram_freq=0,
                          batch_size=BATCH_SIZE)
    tb_disc.set_model(disc)

    for epoch in range(200):
        logger.info("Epoch is %s" % epoch)
        logger.debug("Number of batches %s" % int(X_train.shape[0] / BATCH_SIZE))
        logger.debug("Batch size: %s" % BATCH_SIZE)

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            if index > 0:
                # show debug data only the first time（只在第一次显示调试数据。）.
                logger.setLevel(logging.INFO)
            # random latent vectors. same size of（随机潜在向量。相同大小的），生成随机数
            noise = np.random.normal(size=(BATCH_SIZE, latent_dim))
            # 正常样本
            alexa_domains = X_train[(index * BATCH_SIZE):(index + 1) * BATCH_SIZE]

            logger.debug("domains_batch size:\t%s" % (alexa_domains.shape,))

            # Generating domains from generator（按照生成器的定义生成域名，假样本）
            generated_domains = genr.predict(noise, verbose=0)
            logger.debug("generated domains shape:\t%s" % (generated_domains.shape,))

            # usual trainig mode
            # combined_domains = np.concatenate((domains_batch, generated_domains))
            # labels = np.concatenate([np.ones((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1))]) # 1 = real, 0 = fake
            # labels += 0.05 * np.random.random(labels.shape)

            labels_size = (BATCH_SIZE, 1)
            # ~1 = real. Label Smoothing technique
            #从一个均匀分布（0.9, 1.1)中随机采样，size: 输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出m*n*k个样本，缺省时输出1个值
            #其实就是生成 1 ......
            labels_real = np.random.uniform(0.9, 1.1, size=labels_size)
            #BATCH_SIZE个假标签，与真样本一样数量
            labels_fake = np.zeros(shape=labels_size)  # 0 = fake

            # alternative training mode:
            # 交替训练模型（轮数 为双数训练正常样本alexa，单数训练假样本generated）
            if index % 2 == 0:
                training_domains = alexa_domains
                labels = labels_real
            else:
                training_domains = generated_domains
                labels = labels_fake

            logger.debug("training set shape\t%s" % (training_domains.shape,))
            logger.debug("target shape %s" % (labels.shape,))

            # training discriminator on both alexa and generated domains
            # 真假样本交替训练判别器
            disc.trainable = True
            #disc_history就是d_loss
            disc_history = disc.train_on_batch(training_domains, labels)
            # ##### DOUBLE TRAINING MODE（或者使用分开训练）
            # disc_history1 = disc.train_on_batch(alexa_domains, labels_real)
            # disc_history2 = disc.train_on_batch(generated_domains, labels_fake)
            # disc_history = np.mean([disc_history1, disc_history2])
            # ##########################

            #固定判别器
            disc.trainable = False

            # training generator model inside the adversarial model
            # 训练对抗模型的生成器

            #生成随机数
            noise = np.random.normal(size=(BATCH_SIZE, latent_dim))  # random latent vectors.
            #生成1
            misleading_targets = np.random.uniform(0.9, 1.1, size=labels_size)
            #欺骗对抗模型，进行对抗训练
            gan_history = gan.train_on_batch(noise, misleading_targets)

        # every epoch do this（每轮训练都做这个）
        # 输出日志
        __write_log(callback=tb_gan,
                    names=gan.metrics_names,
                    logs=gan_history,
                    batch_no=epoch)
        __write_log(callback=tb_disc,
                    names=disc.metrics_names,
                    logs=disc_history,
                    batch_no=epoch)
        #每轮都将生成器，判别器，对抗模型的参数保存
        gan.save(os.path.join(directory, 'model/gan.h5'))
        disc.save(os.path.join(directory, 'model/discriminator.h5'))
        genr.save(os.path.join(directory, 'model/generator.h5'))

        #输出本轮的判别器d的损失值
        d_log = ("epoch %d\t[ DISC\tloss : %f ]" % (epoch, disc_history))
        # 输出本轮的对抗模型（即为生成器g）g的损失值
        logger.info("%s\t[ ADV\tloss : %f ]" % (d_log, gan_history))
        #展示生成的域名和矩阵
        generate(generated_domains, n_samples=15, inv_map=data_dict['inv_map'], add_vecs=True)
        #当d_loss太低，已经没有提升的价值，退出
        if float(disc_history) < 0.1:
            logger.error("D loss too low, failure state. terminating...")
            exit(1)

# 生成域名函数
def generate(predictions, inv_map=None, n_samples=5, temperature=1.0, add_vecs=False, save_file=False, model_dir=None):
    #如果为空，重新创建inv_map ：a-z，0-9，还有- . 38个
    if inv_map is None:
        datas_dict = __build_dataset()
        inv_map = datas_dict['inv_map']

    sampled = []
    # generated_domains的数据集合仲提取前n_samples个
    for x in predictions[:n_samples]:
        word = []
        #遍历提取出来的n_samples个数字
        for y in x:
            word.append(__np_sample(y, temperature=temperature))
        #打乱，重组为不同排序的举证
        sampled.append(word)
    # 基于inv_map，和sampled的排序，组合成不同的组合的域名
    readable = __to_readable_domain(np.array(sampled), inv_map=inv_map)
    #将每个域名和该域名对应的矩阵（词向量）打包为列表，并且输出
    if add_vecs:
        import itertools
        for s, r in zip(sampled, readable):
            logger.info("%s\t%s" % (s, r))
    else:
        logger.info("Generated sample: %s " % readable)
    # 将生成的域名保存
    if save_file:
        with open("experiments/%s/samples.txt" % model_dir, 'w') as fp:
            for r in readable:
                if len(r) > 0:
                    fp.write("%s\n" % r)
            print("file saved to %s" % fp)
    return readable

def train_autoencoder():
    data_dict = __build_dataset(n_samples=100000)

    directory = os.path.join("autoencoder_experiments", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(directory):
        # crea la cartella
        os.makedirs(directory)
        os.makedirs(directory + "/weights")

    d = discriminator_model()
    g = generator_model()
    # aenc = Sequential()
    # aenc.add(d)
    # aenc.add(g)

    adv_model = Sequential()
    adv_model.add(g)
    adv_model.add(d)

    adv_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    adv_model.fit(data_dict['X_train'], data_dict['X_train'],
             verbose=2,
             callbacks=[TensorBoard(log_dir=os.path.join(directory, ".logs"),
                                    histogram_freq=0,
                                    write_graph=0),
                        ModelCheckpoint(os.path.join(directory, "weights/autoencoder.h5"),
                                        monitor='val_loss',
                                        verbose=2,
                                        save_best_only=True,
                                        mode='auto')
                        ],
             validation_split=0.33,
             batch_size=128,
             epochs=500)

    adv_model.save(os.path.join(directory, 'aenc_model.h5'))
    print("X_test")
    print(data_dict['X_test'].shape())

    predictions = adv_model.predict(data_dict['X_test'], verbose=0)
    sampled = []
    for x in predictions:
        word = []
        for y in x:
            word.append(__np_sample(y))
        sampled.append(word)

    print("results")
    readable = __to_readable_domain(np.array(sampled), inv_map=data_dict['inv_map'])
    for r in readable:
        print(r)


def test_autoencoder():
    directory = "autoencoder_experiments/20171218-101804"
    data_dict = __build_dataset(n_samples=12000)
    d = discriminator_model()
    g = generator_model()
    # aenc = Sequential()
    # aenc.add(d)
    # aenc.add(g)

    adv_model = Sequential()
    adv_model.add(g)
    adv_model.add(d)
    adv_model.load_weights(os.path.join(directory, 'weights/autoencoder.h5'))

    predictions = adv_model.predict(data_dict['X_train'], verbose=0)
    sampled = []
    for x in predictions:
        word = []
        for y in x:
            word.append(__np_sample(y))
        sampled.append(word)

    print("results")
    readable = __to_readable_domain(np.array(sampled), inv_map=data_dict['inv_map'])
    with open(os.path.join(directory, 'samples.txt'), 'w') as fp:
        for r in readable:
            fp.write("%s\n" % r)


def __custom_gan_loss(y_true, y_pred):
    return -(K.max(K.log(y_pred)))


def __build_dataset(n_samples=10000, maxlen=15, validation_split=0.33):
    #加载正常域名
    df = pd.DataFrame(
        pd.read_csv("resources/datasets/legitdomains.txt",

          # pd.read_csv("../../data/dga/legitdomains.txt",
                    sep=" ",
                    header=None,
                    names=['domain'],
                    ))
    df = df.loc[df['domain'].str.len() > 5]
    # 随机抽取数据集中的一部分，n为要抽取的行数，random_state为随机数发生器种子
    if n_samples:
        df = df.sample(n=n_samples, random_state=42)

    X_ = df['domain'].values
    # preprocessing text（预处理文本）
    tk = Tokenizer(char_level=True)
    tk.fit_on_texts(string.ascii_lowercase + string.digits + '-' + '.')
    seq = tk.texts_to_sequences(X_)
    X = sequence.pad_sequences(seq, maxlen=maxlen)
    #会输出所有词汇与index--》也就是词表【切记如果词汇中包含大写字母，会被转成小写，后面做初始化embedding的时候，切记要转成大写】
    inv_map = {v: k for k, v in tk.word_index.items()}
    X_tmp = []
    #将X的数据转为onehot（二值化0101）
    for x in X:
        X_tmp.append(to_categorical(x, tk.document_count))
    # 标签类别总数
    b =tk.document_count
    X = np.array(X_tmp)
    c = X[int(X.shape[0] * validation_split):, :, :]
    return {'X_train': X[int(X.shape[0] * validation_split):, :, :],
            "X_test": X[:int(X.shape[0] * validation_split), :, :],
            "word_index": tk.document_count,
            "inv_map": inv_map,
            "legit_domain":X_}


def __np_sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array（从概率数组中采样索引的辅助函数）

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    preds = np.random.multinomial(1, preds, 1)
    return np.argmax(preds)


def __sampling(preds, temperature=1.0):
    """
     helper function to sample an index from a probability array

    :param preds: predictions data. 3D tensor of floats
    :param temperature: temperature
    :return: sampled data. 2D tensor of integers
    """
    preds = K.log(preds) / temperature
    exp_preds = K.exp(preds)
    preds = exp_preds / K.sum(exp_preds)
    return K.argmax(preds, axis=2)


def __to_readable_domain(decoded, inv_map):
    domains = []
    for j in range(decoded.shape[0]):
        word = ""
        for i in range(decoded.shape[1]):
            if decoded[j][i] != 0:
                word = word + inv_map[decoded[j][i]]
        domains.append(word)
    return domains


def __write_log(callback, names, logs, batch_no):
    if isinstance(logs, list):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
    else:
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = logs
        summary_value.tag = names[0]
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='moretrain')
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--model", type=str, default='empty')
    parser.add_argument("--save-file", type=bool, default=False)
    args = parser.parse_args()
    return args
def plot_test(legit_domain,sample_data):
    old_d = dict()
    old_d1 =dict()
    for i in range(len(legit_domain)):
        old_d = histogram(legit_domain[i], old_d)
    print(old_d)
    for i in range(len(sample_data)):
        old_d1 = histogram(sample_data[i], old_d1)
    print(old_d1)
    voc_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','_','-']
    new_old_d = []
    new_old_d1 =[]
    c =sorted(old_d)
    # for i in sorted(old_d):
    #     for j in voc_list:
    #         if i == j:
    #             d = old_d[i]
    #         else:
    #             d = 0
    #     new_old_d.append(d)
        # new_old_d.append(old_d[i])
        # new_old_d1.append(old_d1[i])

    for i in voc_list:
        for j in sorted(old_d):
            if i == j:
                d = old_d[i]
                break
            else:
                d = 0
        new_old_d.append(d)

    for i in voc_list:
        for j in sorted(old_d1):
            if i == j:
                d = old_d1[i]
                break
            else:
                d = 0
        new_old_d1.append(d)

    print(voc_list)
    print(new_old_d)
    print(new_old_d1)

    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    x = np.arange(38)
    y = np.array(new_old_d)
    y1 = np.array(new_old_d1)
    bar_width = 0.35
    tick_label = voc_list
    plt.bar(x, y, bar_width, align="center", color="c", label="班级A", alpha=0.5)
    plt.bar(x + bar_width, y1, bar_width, color="b", align="center", label="班级", alpha=0.5)
    plt.xlabel("测试难度")
    plt.ylabel("试卷份数")
    plt.xticks(x + bar_width / 2, tick_label)
    plt.legend()
    plt.show()


def histogram(s, old_d):
  d = old_d
  for c in s:
    d[c] = d.get(c, 0) + 1
  return d



if __name__ == "__main__":
    #从命令行获取参数，默认执行的mode是训练train
    args = get_args()
    #训练模型
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, weights=True)
    if args.mode == "moretrain":
        # model_name = args.model
        model_name ="20171219-235309_BEST"
        disc = load_model("experiments/%s/model/discriminator.h5" % model_name)
        genr = load_model("experiments/%s/model/generator.h5" % model_name)
        train(BATCH_SIZE=args.batch_size,
              disc=disc,
              genr=genr,
              original_model_name=model_name)
    # if args.mode == "plot":
    #     model_name = args.model
        disc = load_model("experiments/%s/model/discriminator.h5" % model_name)
        genr = load_model("experiments/%s/model/generator.h5" % model_name)
        plot_model(disc, "experiments/%s/model/discriminator.png" % model_name, show_shapes=True)
        plot_model(genr, "experiments/%s/model/generator.png" % model_name, show_shapes=True)
    elif args.mode == "generate":
        model_name = "20171219-235309_BEST"
        model = load_model("experiments/%s/model/generator.h5" % model_name)
        preds = model.predict_on_batch(np.random.normal(size=(args.batch_size, 20)))
        legit_domain,sample_data =generate(predictions=preds,n_samples=args.batch_size,add_vecs=False,save_file=args.save_file,model_dir=args.model)
        plot_test(legit_domain,sample_data)
    elif args.mode == "autoencoder":
        train_autoencoder()
    elif args.mode == "test-autoencoder":
        test_autoencoder()
