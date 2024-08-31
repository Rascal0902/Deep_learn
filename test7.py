# softmax는 multinomial classification(다중분류)를 위해 사용

# sigmoid는 cast를 통해 0,1 binary classification만 가능

# soft max로 하더라도 one got encoding으로 binary가능

# softmax의 cost function은 entropy로 유도


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6],
          [1, 7, 7, 7]]

y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder("float", [None, 4])  # shape주의

Y = tf.placeholder("float", [None, 3])

nb_classes = 3  # number of classes

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')  # input : 4, output : nb_classes

b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations

# softmax = exp(logits)/reduce_sum(exp(logits),dim)

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cross entropy cost/loss

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph

# with tf.Session() as sess :

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):

    sess.run(optimizer, feed_dict={X: x_data, Y: y_data})

    if step % 2000 == 0:
        print(step, sess.run([cost, W, b], feed_dict={X: x_data, Y: y_data}))

# Testing & one-hot encoding(arg_max)

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})

print(a, sess.run(tf.arg_max(a, 1)))

print(a)