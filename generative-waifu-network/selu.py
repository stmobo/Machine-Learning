import tensorflow as tf
import numpy as np

# Scaled Exponential Linear Unit
# Lambda-01 and Alpha-01 are taken from the paper
# [arxiv: 1706.02515]
def scaled_elu(tensor_in, lm=1.0507, al=1.6733):
    cond = tf.greater(tensor_in, 0.0)
    neg = al * (tf.exp(tensor_in) - 1.0)

    return lm * tf.where(cond, tensor_in, neg)

def alpha_dropout(tensor_in, drop_rate=0.05, lm=1.0507, al=1.6733):
    if drop_rate == 0:
        return tensor_in

    q = 1.0 - drop_rate # drop_rate = 1 - q
    dropout_dist = tf.contrib.distributions.Bernoulli(probs=q, dtype=tf.float32)
    dropout_var = dropout_dist.sample(tf.shape(tensor_in))

    al_prime = -lm * al

    # Randomly set activations to al_prime
    ret = (tensor_in * dropout_var) + (al_prime * (1.0 - dropout_var))

    # Adjust activation mean and variance
    a = np.pow(q + (np.pow(al_prime, 2) * q * (1.0-q)), -0.5)
    b = -a * (1.0 - q) * al_prime
    return (a * ret) + b

# Also: Initialize weights from a gaussian distribution
# With mean 0 and variance 1/n (where n = number of parameters)
def initializer(shape, dtype=tf.float32, partition_info=None):
    shape_list = shape.as_list()
    n = 1 # number of units
    for dim in shape_list:
        n *= dim

    # return gaussian distribution, with
    # E(w_i) = 0
    # Var(w_i) = 1/n
    return tf.random_normal(shape, mean=0.0, stddev=np.sqrt(np.reciprocal(n)))
