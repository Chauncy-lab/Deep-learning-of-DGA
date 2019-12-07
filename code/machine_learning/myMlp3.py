import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("../../data/mnist/",one_hot=True)

sess=tf.InteractiveSession()

in_units=784

h0_units=300

h1_units=200

h2_units=100

w0=tf.Variable(tf.random_normal([in_units,h0_units],stddev=0.1))

w1=tf.Variable(tf.random_normal([h0_units,h1_units],stddev=0.1))

b0=tf.Variable(tf.zeros([h0_units]))

b1=tf.Variable(tf.zeros([h1_units]))


w2=tf.Variable(tf.random_normal([h1_units,h2_units],stddev=0.1))

b2=tf.Variable(tf.zeros([h2_units]))


w3=tf.Variable(tf.zeros([h2_units,10]))

b3=tf.Variable(tf.zeros([10]))

x=tf.placeholder(tf.float32,[None,in_units])

keep_prob=tf.placeholder(tf.float32)


hidden0=tf.nn.relu(tf.matmul(x,w0)+b0)

hidden0_drop=tf.nn.dropout(hidden0,keep_prob)

hidden1=tf.nn.relu(tf.matmul(hidden0,w1)+b1)
#
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)


hidden2=tf.nn.relu(tf.matmul(hidden1,w2)+b2)
#
hidden2_drop=tf.nn.dropout(hidden2,keep_prob)



y=tf.nn.softmax(tf.matmul(hidden2,w3)+b3)
#
y_=tf.placeholder(tf.float32,[None,10])

corss_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

train=tf.train.AdagradOptimizer(0.1).minimize(corss_entropy)

tf.global_variables_initializer().run()



for i in range(3000):

    batch_xs,batch_ys=mnist.train.next_batch(100)

    train.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
