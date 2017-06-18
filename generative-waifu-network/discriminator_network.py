import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import selu

def conv_size_divide(dim, div):
    return int(np.ceil(float(dim) / float(div)))

class DiscriminatorNetwork:
    def __init__(self, args, image_in, labels_in):
        self.args = args
        self.image = image_in
        self.labels = labels_in

    def inception_module(self, tensor_in, scope='Inception'):
        batch_size, input_height, input_width, input_depth = tensor_in.shape.as_list()
        output_height, output_width = conv_size_divide(input_height, 2), conv_size_divide(input_width, 2)

        bottleneck_depth = self.args.dsc_bottleneck_depth
        output_depth = self.args.dsc_layer_depth
        head_depth = self.args.head_depth // 4

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
        return out

    def conv_layer(self, tensor_in, scope='Conv'):
        batch_size, input_height, input_width, input_depth = tensor_in.shape.as_list()
        output_height, output_width = conv_size_divide(input_height, 2), conv_size_divide(input_width, 2)
        output_depth = self.args.dsc_layer_depth

        net = slim.conv2d(tensor_in, output_depth, kernel_size=3, stride=2, scope=scope)
        net = tf.reshape(net, [batch_size, output_height, output_width, output_depth])

        return net
