import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]

y_data = [[1], [1], [1], [0], [0], [0]]

# placeholder을 만들때는 shape에 주의

X = tf.placeholder(tf.float32, shape=[None, 2])

Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')

b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis를 sigmoid에 통과 : 0~1의 값으로 나오게 될것임.

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# sigmoid를 통과한 값이 0.5보다 크면 1 아니면 0 으로 기준 설정(Cast)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)  # True = 1, False = 0

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())  # Variable 초기화

for step in range(5001):

    cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})

    if step % 1000 == 0:
        print("%05d" % step, cost_val)

h, c, a, w, b, cost = sess.run([hypothesis, predicted, accuracy, W, b, cost],

                               feed_dict={X: x_data, Y: y_data})

print("\n [5001번 학습결과]")

# print(h)

print("1. 시그모이드 적용 : ", h.T)

print("2. + 활성함수 적용 : ", [int(x) for x in c])

print("3. 기존 정답과 비교 : ", sum(y_data, []))

print("정확도(Accuracy) : ", a)

print("Weighting : ", w)

print("bias :", b)

print("cost : ", cost)