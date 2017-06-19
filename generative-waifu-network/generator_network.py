import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import selu

def leaky_relu(tensor_in):
    return tf.maximum(tensor_in, tensor_in * 0.01)

def generator_parameters(parser):
    parser.add_argument('--z-size', type=int, default=256, help='Dimensionality of Z (noise) vectors')
    parser.add_argument('--label-size', type=int, default=1000, help='Dimensionality of Y (tag) vectors')

    parser.add_argument('--deception-modules', action='store_true', help='Network will be built from Deception modules if true')

    parser.add_argument('--output-height', type=int, default=1024, help='Height of output images')
    parser.add_argument('--output-width', type=int, default=1024, help='Width of output images')

    parser.add_argument('--gen-final-depth', type=int, default=64, help='Final number of generator deconv filters (before image output layer)')
    parser.add_argument('--gen-bottleneck-depth', type=int, default=64, help='Bottleneck layer depth for Deception modules')

    parser.add_argument('--gen-layers', type=int, default=4, help='Number of generator deconv layers (not including output layer)')
    parser.add_argument('--gen-kernel-size', type=int, default=5, help='Height+Width of generator deconv layer kernels')

class Generator:
    def __init__(self, args, noise_tensor, labels_tensor):
        self.args = args
        self.noise = noise_tensor
        self.labels = labels_tensor

    def build(self, scope='Generator', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            net = self.input_layer(self.noise, self.labels)

            for layer in range(self.args.gen_layers):
                current_layer_depth = self.args.gen_final_depth // (2 ** (self.args.gen_layers - layer - 1))

                if self.args.deception_modules:
                    net = self.deception_module(net, current_layer_depth, scope='Deception{:d}'.format(layer))
                else:
                    net = self.standard_deconv_layer(net, current_layer_depth, scope='ConvT{:d}'.format(layer))

            self.out = self.output_layer(net)
            
        self.vars = slim.get_trainable_variables(scope=scope)
        return self.out

    def activation_fn(self):
        if self.args.gen_activation == 'relu':
            return tf.nn.relu
        elif self.args.gen_activation == 'lrelu':
            return leaky_relu
        elif self.args.gen_activation == 'selu':
            return selu.selu
        raise ValueError("Invalid value for --gen-activation: " + self.args.gen_activation)

    def initializer_fn(self):
        if self.args.gen_activation == 'relu' or self.args.gen_activation == 'lrelu':
            return None
        else:
            return selu.initializer

    def network_arg_scope(self):
        return slim.arg_scope([slim.fully_connected, slim.conv2d, slim.conv2d_transpose], activation_fn=self.activation_fn(), weights_initializer=self.initializer_fn())

    def input_layer(self, noise_in, labels_in, scope='Input'):
        input_layer_divisor = int(2 ** args.gen_layers)

        input_layer_height = self.args.output_height // input_layer_divisor
        input_layer_width = self.args.output_width // input_layer_divisor
        input_layer_depth = self.args.gen_final_depth // input_layer_divisor

        batch_size = noise_in.shape.as_list()[0]
        projection_units = input_layer_height * input_layer_width

        with tf.variable_scope(scope):
            with slim.arg_scope(self.network_arg_scope()):
                projected_noise = slim.fully_connected(noise_in, projection_units, scope='noise-projection')
                projected_labels = slim.fully_connected(labels_in, projection_units, scope='label-projection')

                projected_noise = tf.reshape(projected_noise, [batch_size, input_layer_height, input_layer_width, 1])
                projected_labels = tf.reshape(projected_labels, [batch_size, input_layer_height, input_layer_width, 1])

                input_layer = tf.concat([projected_noise, projected_labels], axis=3)

                # Combine noise and label features
                net = tf.conv2d(input_layer, input_layer_depth,
                    kernel_size=1,
                    scope='input-conv'
                )

                return net

    def deception_module(self, tensor_in, output_depth, scope='Deception'):
        batch_size, layer_height, layer_width, layer_depth = tensor_in.shape.as_list()
        output_height, output_width = layer_height * 2, layer_width * 2
        head_depth = output_depth // 4

        bottleneck_depth = self.args.gen_bottleneck_depth

        with tf.variable_scope(scope):
            with slim.arg_scope(self.network_arg_scope()):
                # 1x1 bottleneck layers + initial scaling
                head_3x3 = slim.conv2d(tensor_in, bottleneck_depth, kernel_size=1, scope='Bottleneck_3x3')
                head_5x5 = slim.conv2d(tensor_in, bottleneck_depth, kernel_size=1, scope='Bottleneck_5x5')
                head_scale = tf.image.resize_nearest_neighbor(tensor_in, (output_height, output_width), scope='Resize_2x')

                # Transposed conv layers and scale head bottlenecking
                head_1x1 = slim.conv2d_transpose(tensor_in, head_depth, kernel_size=1, stride=2, scope='ConvT_1x1')
                head_3x3 = slim.conv2d_transpose(head_3x3, head_depth, kernel_size=3, stride=2, scope='ConvT_3x3')
                head_5x5 = slim.conv2d_transpose(head_5x5, head_depth, kernel_size=5, stride=2, scope='ConvT_5x5')
                head_scale = slim.conv2d(head_scale, head_depth, kernel_size=1, scope='Bottleneck_2x')

                # Reshape and depth-concatenate heads
                head_1x1 = tf.reshape(head_1x1, [batch_size, output_height, output_width, head_depth])
                head_3x3 = tf.reshape(head_3x3, [batch_size, output_height, output_width, head_depth])
                head_5x5 = tf.reshape(head_5x5, [batch_size, output_height, output_width, head_depth])
                head_scale = tf.reshape(head_scale, [batch_size, output_height, output_width, head_depth])

                out = tf.concat([head_1x1, head_3x3, head_5x5, head_scale], axis=3)
                out = tf.reshape(out, [batch_size, output_height, output_width, output_depth])

                return out

    def standard_deconv_layer(self, tensor_in, output_depth, scope='ConvT'):
        batch_size, layer_height, layer_width, layer_depth = tensor_in.shape.as_list()
        output_height, output_width = layer_height * 2, layer_width * 2

        with slim.arg_scope(self.network_arg_scope()):
            out = slim.conv2d_transpose(tensor_in, output_depth,
                kernel_size=self.args.gen_kernel_size,
                stride=2,
                normalizer_fn=slim.fused_batch_norm
                scope=scope
            )
            out = tf.reshape(out, [batch_size, output_height, output_width, output_depth])

        return out

    def output_layer(self, tensor_in, scope='Output'):
        net = slim.conv2d(tensor_in, 3,
            kernel_size=self.args.gen_kernel_size,
            activation_fn=tf.tanh,
            scope=scope
        )
        out = tf.reshape(net, [-1, self.args.output_height, self.args.output_width, 3])

        return out
