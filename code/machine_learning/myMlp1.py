import tensorflow as tf
import pickle
import gzip

def get_one_hot(x,size=10):
    v=[]
    for x1 in x:
        x2=[0]*size
        x2[(x1-1)]=1
        v.append(x2)
    return v

def load_data():
    with gzip.open('../../data/MNIST/mnist.pkl.gz') as fp:
        training_data, valid_data, test_data = pickle.load(fp,encoding='bytes')
    return training_data, valid_data, test_data


if __name__ == "__main__":

    training_data, valid_data, test_dat=load_data() #加载数据
    x_training_data,y_training_data=training_data #将训练集（囊括了原数据和标签集）拆分开
    x1_test_data,y1_test_data=test_dat#将测试集（囊括了原数据和标签集）拆分开

    y_training_data=get_one_hot(y_training_data) #one-hot编码处理标记数据
    y1_test_data=get_one_hot(y1_test_data)


    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y_ = tf.placeholder("float", [None, 10])

    batch_size=100
    y = tf.nn.softmax(tf.matmul(x,W) + b)


    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(int(len(x_training_data)/batch_size)):
        batch_xs=x_training_data[(i*batch_size):((i+1)*batch_size)]
        batch_ys=y_training_data[(i*batch_size):((i+1)*batch_size)]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) #比较在向量行中找最大值的索引,是否一致
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: x1_test_data, y_: y1_test_data}))