import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import matplotlib.pyplot as plt

import random

tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

# Check out https://www.tensorflow.org/get_started/mnist/beginners for

# more information about the mnist dataset

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784

X = tf.placeholder(tf.float32, [None, 784])

# 0 - 9 digits recognition = 10 classes

Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))

b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

# Calculate accuracy

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters

num_epochs = 15

batch_size = 100

num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    # Initialize TensorFlow variables

    sess.run(tf.global_variables_initializer())

    # Training cycle

    for epoch in range(num_epochs):

        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})

            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test the model using test sets

    print(

        "Accuracy: ",

        accuracy.eval(

            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}

        ),

    )

    # Get one and predict

    r = random.randint(0, mnist.test.num_examples - 1)

    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r: r + 1], 1)))

    print(

        "Prediction: ",

        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r: r + 1]}),

    )

    plt.imshow(

        mnist.test.images[r: r + 1].reshape(28, 28),

        cmap="Greys",

        interpolation="nearest",

    )

    plt.show()