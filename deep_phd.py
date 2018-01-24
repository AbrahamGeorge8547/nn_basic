import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
init = tf.initialize_all_variables()
y = tf.nn.softmax(tf.matmul(tf.reshape(x, [-1, 784]), W) + b)
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_true*tf.log(y))
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_data = {x: batch_x, y_true: batch_y}
    sess.run(train_step, feed_dict=train_data)
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    if i % 100 == 0:
        test_data = {x: mnist.test.images, y_true: mnist.test.labels}
        print(sess.run(accuracy, feed_dict=test_data))





