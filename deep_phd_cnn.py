import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
k = 6
l = 12
m = 24
def new_weights(a, b, c, d):
    w = tf.Variable(tf.truncated_normal([a, b, c, d], stddev=0.1))
    b_ = tf.Variable(tf.ones([d])/10)
    return w, b_

w1, b1 = new_weights(5, 5, 1, k)
w2, b2 = new_weights(4, 4, k, l)
w3, b3 = new_weights(4, 4, l, m)

n = 200
w4 = tf.Variable(tf.truncated_normal([7*7*m, n], stddev=0.1))
b4 = tf.Variable(tf.ones([200])/10)
w5 = tf.Variable(tf.truncated_normal([n, 10], stddev=0.1))
b5 = tf.Variable(tf.ones([10])/10)

x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
lr = tf.placeholder(tf.float32)
y1 = tf.nn.relu(tf.nn.conv2d(x_image, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[1, 2, 2, 1], padding='SAME') + b3)
yy = tf.reshape(y3, shape=[-1, 7*7*m])
yp = tf.nn.relu(tf.matmul(yy, w4) + b4)
y4 = tf.nn.dropout(yp, 0.75)
y = tf.nn.softmax(tf.matmul(y4, w5) + b5)
sess = tf.Session()
init = tf.global_variables_initializer()
true = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
accuracy = tf.reduce_mean(tf.cast(true, tf.float32))
optmizer = tf.train.GradientDescentOptimizer(lr)
train_step = optmizer.minimize(cross_entropy)
sess.run(init)
max_learning_rate = 0.0003
min_learning_rate = 0.0001
decay_speed = 2000.0
for i in range(8000):
    batch_x, batch_y = mnist.train.next_batch(100)
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    train_data = {x: batch_x, y_: batch_y, lr: learning_rate}
    sess.run(train_step, feed_dict=train_data)
    if i % 100 == 0:
        test_data = {x: mnist.test.images, y_: mnist.test.labels}
        print(sess.run(accuracy, feed_dict=test_data))
