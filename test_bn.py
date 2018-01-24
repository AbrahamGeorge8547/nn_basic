import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
k = 24
l = 48
m = 64
n = 200
tf.set_random_seed(0.0)
# convolution layer


def new_weights(a, b, c, d):
    w = tf.Variable(tf.truncated_normal([a, b, c, d], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, tf.float32, [d]))
    return w, b


w1, b1 = new_weights(6, 6, 1, k)
w2, b2 = new_weights(5, 5, k, l)
w3, b3 = new_weights(4, 4, l, m)

# fully connected layer

w4 = tf.Variable(tf.truncated_normal([7*7*m, n], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, tf.float32, [n]))
w5 = tf.Variable(tf.Variable(tf.truncated_normal([n, 10], stddev=0.1)))
b5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))


# placer holders

X = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(X, [-1, 28, 28, 1])

Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# batch norm function


def batchnorm(Ylogits, is_test, iteration, offset, con=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    epsilon = 1e-5
    if con:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, epsilon)
    return ybn, update_moving_averages


# layer 1
y1l = tf.nn.conv2d(x_image, w1, strides=[1, 1, 1, 1], padding='SAME')
y1bn, uma1 = batchnorm(y1l, tst, iter, b1, con=True)
y1 = tf.nn.relu(y1bn)

# layer2

y2l = tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME')
y2bn, uma2 = batchnorm(y2l, tst, iter, b2, con=True)
y2 = tf.nn.relu(y2bn)

# layer 3

y3l = tf.nn.conv2d(y2, w3, strides=[1, 2, 2, 1], padding='SAME')
y3bn, uma3  = batchnorm(y3l, tst, iter, b3, con =True)
y3 = tf.nn.relu(y3bn)

# reshape

yy = tf.reshape(y3, shape=[-1,7*7*m])

y4l = tf.matmul(yy,w4)
y4bn, uma4 = batchnorm(y4l, tst, iter, b4)
y4 = tf.nn.relu(y4bn)

Y5 = tf.matmul(y4, w5) + b5
Y = tf.nn.softmax(Y5)


update_ema = tf.group(uma1, uma2, uma3, uma4)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y5, labels= Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(8000):
    batch_x, batch_y = mnist.train.next_batch(100)
    max_lr = 0.02
    min_lr = 0.0001
    decay_speed = 1600
    learning_rate = min_lr + (max_lr-min_lr)*math.exp(-i/decay_speed)
    train_data = {X: batch_x, Y_: batch_y, lr: learning_rate, tst: False}
    train_data1 = {X: batch_x, Y_: batch_y, lr: learning_rate, tst: False, iter: i}
    sess.run(train_step, feed_dict=train_data)
    sess.run(update_ema, feed_dict=train_data1)
    if i % 100 == 0:
        test_data = {X: mnist.test.images, Y_: mnist.test.labels, tst: True }
        print(sess.run(accuracy, feed_dict=test_data))
























