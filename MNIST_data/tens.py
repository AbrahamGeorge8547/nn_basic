import tensorflow as tf
import numpy as np


# input data
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3
b = tf.Variable(tf.zeros(1))
w = tf.Variable(tf.random_uniform([1, 2], -1, 1))
y = tf.matmul(w,x_data) + b

# GD

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


# initlise  Variable
init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)

for step in xrange(200):
    sess.run(train)
print sess.run(w)



