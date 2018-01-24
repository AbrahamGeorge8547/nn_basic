import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
k = 200
l = 100
m = 60
n = 30
def new_weights(k1, l):
    w = tf.Variable(tf.truncated_normal((k1, l), stddev=0.1))
    b = tf.Variable(tf.zeros([l]))
    return w, b


def sigmoid(x, w, b):
    return tf.nn.relu(tf.matmul(x, w) + b)


w1, b1 = new_weights(784, k)
w2, b2 = new_weights(k, l)
w3, b3 = new_weights(l, m)
w4, b4 = new_weights(m, n)
w5, b5 = new_weights(n, 10)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
tf.reshape(x, [-1, 784])
y1 = sigmoid(x, w1, b1)
y2 = sigmoid(y1, w2, b2)
y3 = sigmoid(y2, w3, b3)
y4 = sigmoid(y3, w4, b4)
y = tf.nn.softmax(tf.matmul(y4, w5) + b5)
init = tf.initialize_all_variables()
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)
sess = tf.Session()
sess.run(init)

# loop
for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_data = {x: batch_x, y_: batch_y}
    sess.run(train_step, feed_dict=train_data)
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    if i % 100 == 0:
        test_data = {x: mnist.test.images, y_: mnist.test.labels}
        print(sess.run(accuracy, feed_dict=test_data))









