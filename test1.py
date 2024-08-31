import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

hello = tf.constant("Hello tensorflow")
sess = tf.Session()

print(sess.run(hello))