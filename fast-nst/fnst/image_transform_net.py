import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import configargparse
import sys

eps = np.finfo(np.float32).eps

# norm_input: input tensor
# i: layer number
# See Ulyanov et al.
def instance_normalization(net):
    mu, sig2 = tf.nn.moments(net, axes=[1,2], keep_dims=True)

    depth = net.shape.as_list()[3]

    scaling = slim.model_variable('scale', shape=[depth], initializer=tf.ones_initializer(), trainable=True)
    shift = slim.model_variable('shift', shape=[depth], initializer=tf.zeros_initializer(), trainable=True)

    out = (net - mu) / tf.sqrt(sig2 + eps)
    return (out * scaling) + shift

# net: res-block input tensor
# block_number: residual block number
def residual_block(block_input, block_number, args):
    #print("TransformNet: res-block {} input shape {}".format(block_number, str(block_input.shape)))
    #sys.stdout.flush()

    with tf.variable_scope('res{}'.format(block_number)):
        net = slim.conv2d(block_input, args.res_block_depth, kernel_size=3, scope='conv1', normalizer_fn=instance_normalization, activation_fn=tf.nn.relu)
        net = slim.conv2d(net, args.res_block_depth, kernel_size=3, scope='conv2', normalizer_fn=instance_normalization, activation_fn=tf.nn.relu)
        return net + block_input

# Performs downsampling with stride-2 convolutions
def input_block(block_input, args):
    print("TransformNet in-block: input shape {}".format(str(block_input.shape)))
    sys.stdout.flush()

    n_downsample_layers = int(np.log2(args.downsampling_factor))
    with tf.variable_scope('input'):
        net = slim.conv2d(block_input, args.res_block_depth // (2**n_downsample_layers), kernel_size=9, stride=1, scope='conv1', normalizer_fn=instance_normalization, activation_fn=tf.nn.relu)
        for i in range(n_downsample_layers):
            depth_divisor = 2**(n_downsample_layers-i-1)
            net = slim.conv2d(
                net,
                args.res_block_depth // depth_divisor,
                kernel_size=3,
                stride=2,
                scope='conv{}'.format(i+2),
                normalizer_fn=instance_normalization,
                activation_fn=tf.nn.relu
            )

        print("TransformNet in-block: output shape {}".format(str(net.shape)))
        sys.stdout.flush()

        return net

# Performs upsampling with rate-2 activations and a final scaled tanh activation
def output_block(block_input, args):
    print("TransformNet out-block: input shape {}".format(str(block_input.shape)))
    sys.stdout.flush()

    n_upsample_layers = int(np.log2(args.upsampling_factor))
    with tf.variable_scope('output'):
        net = block_input

        for i in range(n_upsample_layers):
            depth_divisor = 2**(i+1)
            net = slim.conv2d_transpose(
                net,
                args.res_block_depth // depth_divisor,
                kernel_size=3,
                stride=2,
                scope='tconv{}'.format(i+1),
                normalizer_fn=instance_normalization,
                activation_fn=tf.nn.relu
            )

        net = slim.conv2d(net, 3, kernel_size=9, stride=1, scope='conv{}'.format(n_upsample_layers+1), normalizer_fn=instance_normalization, activation_fn=tf.tanh)
        net *= args.tanh_factor
        net += (255.0 / 2)
        #net = tf.clip_by_value(net, 0, 255.0)

        print("TransformNet out-block: output shape {}".format(str(net.shape)))
        sys.stdout.flush()

        return net

def add_network_hyperparameters(parser):
    parser.add_argument('--network-scope', default='nst-model', help='Variable scope to use for network parameters.')
    parser.add_argument('--res-block-depth', type=int, default=128, help='Convolutional layer depth within residual blocks.')
    parser.add_argument('--n-res-blocks', type=int, default=5, help='Number of residual blocks to create in network.')
    parser.add_argument('--downsampling-factor', type=float, default=4.0, help='Spatial downsampling factor for network inputs. Will be rounded to a power of two.')
    parser.add_argument('--upsampling-factor', type=float, default=4.0, help='Spatial upsampling factor for network outputs. Will be rounded to a power of two.')
    parser.add_argument('--tanh-factor', type=float, default=150.0, help='Final network outputs are multiplied by this factor (for converting from the tanh output range to normal image pixel values).')

def network_parameters(args):
    return slim.get_model_variables(scope=args.network_scope)

# Builds the image transformation network graph.
# The image input tensor (image_in) should be 4D w/ shape [batch, height, width, channels].
# The output tensor is of the same shape.
def transform_net(image_in, args):
    with tf.variable_scope(args.network_scope):
        net = input_block(image_in, args)

        for res_block_i in range(args.n_res_blocks):
            net = residual_block(net, res_block_i+1, args)

        return output_block(net, args)
