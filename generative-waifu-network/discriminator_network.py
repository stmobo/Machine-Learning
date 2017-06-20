import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import selu

from common import *

def discriminator_parameters(parser):
    group = parser.add_argument_group('Discriminator Network Parameters', description='These parameters control the structure of the discriminator / critic network.')

    group.add_argument('--inception-modules', action='store_true', help='Discriminator network will be built from Inception modules if true')
    group.add_argument('--dsc-final-depth', type=int, default=64, help='Final number of discirminator conv filters (before image output layer)')
    group.add_argument('--dsc-bottleneck-depth', type=int, default=64, help='Bottleneck layer depth for Inception modules')
    group.add_argument('--dsc-layers', type=int, default=4, help='Number of discriminator conv layers (not including output layer)')
    group.add_argument('--dsc-activation', help='Activation function to use for discriminator network')
    group.add_argument('--dsc-weight-clip', type=float, default=1e-2, help='Critic network weight clipping values (for Wasserstein GANs)')
    group.add_argument('--dsc-normalizer', help='Normalization function to use in network layers (valid values are \'batch\', \'layer\', \'none\')')

class Discriminator:
    def __init__(self, args, image_in):
        self.args = args
        self.image = image_in

    def build(self, scope='Discriminator', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            net = self.input_layer(self.image, self.labels)

            for layer in range(self.args.dsc_layers):
                current_layer_depth = self.args.dsc_final_depth // (2 ** (self.args.dsc_layers - layer - 1))

                if self.args.inception_modules:
                    net = self.inception_module(net, current_layer_depth, scope='Inception{:d}'.format(layer))
                else:
                    net = self.conv_layer(net, current_layer_depth, scope='Conv{:d}'.format(layer))

            self.critic, self.pred_labels = self.output_layer(net)

        self.vars = slim.get_trainable_variables(scope=scope)
        return self.critic, self.pred_labels

    def activation_fn(self):
        if self.args.dsc_activation == 'relu':
            return tf.nn.relu
        elif self.args.dsc_activation == 'lrelu':
            return leaky_relu
        elif self.args.dsc_activation == 'selu':
            return selu.selu
        raise ValueError("Invalid value for --dsc-activation: " + self.args.dsc_activation)

    def initializer_fn(self):
        if self.args.dsc_activation == 'relu' or self.args.dsc_activation == 'lrelu':
            return None
        else:
            return selu.initializer

    def normalizer_fn(self):
        if self.args.dsc_normalizer == 'batch':
            return slim.fused_batch_norm
        elif self.args.dsc_normalizer == 'layer':
            return slim.layer_norm
        elif self.args.dsc_normalizer == 'none':
            return None
        raise ValueError("Invalid value for --dsc-normalizer: " + self.args.dsc_normalizer)

    def network_arg_scope(self):
        return slim.arg_scope([slim.fully_connected, slim.conv2d, slim.conv2d_transpose], activation_fn=self.activation_fn(), weights_initializer=self.initializer_fn())

    def clip_network_weights(self, apply_grad_ops):
        with tf.control_dependencies([apply_grad_ops]):
            clipped_weights = [tf.clip_by_value(w, -args.dsc_weight_clip, args.dsc_weight_clip) for w in self.vars]
            return tf.group(*clipped_weights)

    def input_gradient_norms(self):
        input_grads = tf.gradients(self.out, self.image, name='dsc-input-gradients')
        return tf.norm(input_grads, axis=(1,2)) # Note: Compute norms over height and width (axes 1,2 for NHWC and 2,3 for NCHW)

    def inception_module(self, tensor_in, output_depth, scope='Inception'):
        batch_size, input_height, input_width, input_depth = tensor_in.shape.as_list()
        output_height, output_width = conv_output_size(input_height, 2), conv_output_size(input_width, 2)

        bottleneck_depth = self.args.dsc_bottleneck_depth
        head_depth = self.args.head_depth // 4

        with slim.arg_scope(self.network_arg_scope()):
            head_pool = slim.avg_pool2d(tensor_in, kernel_size=3, stride=2, padding='SAME', scope='AvgPool')
            head_1x1 = slim.conv2d(tensor_in, head_depth, kernel_size=1, stride=2, scope='Conv1x1')
            head_3x3 = slim.conv2d(tensor_in, bottleneck_depth, kernel_size=1, scope='Conv3x3-BN')
            head_5x5 = slim.conv2d(tensor_in, bottleneck_depth, kernel_size=1, scope='Conv5x5-BN')

            head_pool = slim.conv2d(head_pool, head_depth, kernel_size=1, scope='AvgPool-BN')
            head_3x3 = slim.conv2d(head_3x3, head_depth, kernel_size=3, scope='Conv3x3')

            # 5x5 conv as stacked 3x3 convolutions
            head_5x5 = slim.conv2d(head_5x5, head_depth, kernel_size=3, scope='Conv5x5-1')
            head_5x5 = slim.conv2d(head_5x5, head_depth, kernel_size=3, scope='Conv5x5-2')

            head_pool = tf.reshape(head_pool, [batch_size, output_height, output_width, head_depth])
            head_1x1 = tf.reshape(head_1x1, [batch_size, output_height, output_width, head_depth])
            head_3x3 = tf.reshape(head_3x3, [batch_size, output_height, output_width, head_depth])
            head_5x5 = tf.reshape(head_5x5, [batch_size, output_height, output_width, head_depth])

            out = tf.concat([head_pool, head_1x1, head_3x3, head_5x5], axis=3)

            norm_fn = self.normalizer_fn()
            if norm_fn is not None:
                out = norm_fn(out)

            return out

    def conv_layer(self, tensor_in, output_depth, scope='Conv'):
        batch_size, input_height, input_width, input_depth = tensor_in.shape.as_list()
        output_height, output_width = conv_output_size(input_height, 2), conv_output_size(input_width, 2)

        with slim.arg_scope(self.network_arg_scope()):
            net = slim.conv2d(tensor_in, output_depth, kernel_size=3, stride=2, normalizer_fn=self.normalizer_fn(), scope=scope)
            net = tf.reshape(net, [batch_size, output_height, output_width, output_depth])

            return net

    def input_layer(self, image_in, labels_in):
        batch_size, input_height, input_width, input_depth = image_in.shape.as_list()
        output_height, output_width = conv_output_size(input_height, 2), conv_output_size(input_width, 2)
        output_depth = self.args.dsc_final_depth // (2 ** self.args.dsc_layers)

        with slim.arg_scope(self.network_arg_scope()):
            net = slim.conv2d(image_in, output_depth, kernel_size=3, scope='Input-Conv1')
            net = slim.conv2d(net, output_depth, kernel_size=3, stride=2, scope='Input-Conv2')

            # projected_labels = slim.fully_connected(labels_in, output_height * output_width, scope='LabelProjection')
            # projected_labels = tf.reshape(projected_labels, [batch_size, output_height, output_width, 1])

            # net = tf.concat([net, projected_labels], axis=3)

            return net

    def output_layer(self, tensor_in):
        batch_size, input_height, input_width, input_depth = image_in.shape.as_list()
        flat_in = tf.reshape(tensor_in, [batch_size, -1])

        critic_out = slim.fully_connected(
            flat_in,
            1,
            activation_fn=None,
            scope='Critic',
        )

        labels_out = slim.fully_connected(
            flat_in,
            self.args.label_size,
            activation_fn=None,
            scope='PredLabels',
        )

        return critic_out, labels_out
