import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np
import scipy as sp

def conv_output_size(dim_in, stride):
    return int(math.ceil(float(dim_in) / float(stride)))

def leaky_relu(tensor_in):
    return tf.maximum(tensor_in, tensor_in * 0.01)

def dsc_image_preprocessing(args, img_file_in):
    img_8u = tf.image.decode_png(img_file_in)
    img_f32 = tf.image.convert_image_dtype(img_8u, tf.float32)

    # Resize all images to match the generator's output dimensions
    img_resized = tf.image.resize_bicubic(img_f32, (args.output_height, args.output_width))

    # Scale all image values from range [0,1] to range [-1, 1] (same as TanH)
    img_out = (img_resized * 2.0) - 1.0

    return img_out

def gen_image_processing(gan_out):
    # Scale image values from [-1, 1] to [0, 1] (TanH -> TF float32 image ranges)
    img_f32 = (gan_out + 1.0) / 2.0
    img_8u = tf.image.convert_image_dtype(img_f32, tf.uint8, saturate=True)

    return img_8u

def waifunet_parameters(parser):
    parser.add_argument('z-size', type=int, default=256, help='Dimensionality of Z (noise) vectors')
    parser.add_argument('label-size', type=int, default=1000, help='Dimensionality of Y (tag) vectors')
    parser.add_argument('output-height', type=int, default=1024, help='Height of output images')
    parser.add_argument('output-width', type=int, default=1024, help='Width of output images')
    parser.add_argument('gen-filter-base', type=int, default=64, help='Final (base) number of generator deconv filters (before image output layer)')
    parser.add_argument('gen-layers', type=int, default=4, help='Number of generator deconv layers (not including output layer)')
    parser.add_argument('gen-kernel-size', type=int, default=5, help='Height+Width of generator deconv layer kernels')
    parser.add_argument('dsc-kernel-size', type=int, default=5, help='Height+Width of discriminator conv layer kernels')
    parser.add_argument('dsc-filter-base', type=int, default=64, help='Initial number of input filters for discriminator network')
    parser.add_argument('dsc-layers', type=int, default=5, help='Number of conv layers in discriminator network (not including output flattening + sigmoid FC output layer)')

    parser.add_argument('learning-rate', type=float, default=2e-4, help='Learning rate for generator and discriminator networks')
    parser.add_argument('beta1', type=float, default=0.5, help='Beta1 parameter for Adam optimizers (for both generator and discriminator)')

    parser.add_argument('gen-use-lrelu', action='store_true', help='Generator network uses Leaky ReLU activations in place of standard ReLU')

    return parser

class waifunet(object):
    # Builds the model.
    # Options:
    # 'z_size' [default 256]: Length / dimensionality of Z vector
    # 'label_size' [default 1000]: Length / dimensionality of Y vector
    # 'output_width', 'output_height' [default 1024 for both]: Dimension of generator output
    # 'gen_filter_base' [default 64]: Final number of deconv filters (before output image layer)
    # 'gen_layers' [default 4]: Number of deconv / frac-strided conv layers in the generator network (not including image output layer)
    # 'gen_kernel_size', 'dsc_kernel_size' [default 5]: size of conv / deconv filter kernels
    # 'dsc_filter_base' [default 64]: Initial number of conv filters
    # 'dsc_layers' [default 5]: Number of conv layers (not including output flattening and sigmoid)
    #
    # 'learning_rate' [recommended value 2e-4]: Learning rate for Adam optimization
    # 'beta1' [recommended value 0.5]: Beta1 parameter for Adam optimization
    #
    # Input tensor parameters:
    # 'noise_in', 'labels_in': Generator network inputs (noise vectors and tags)
    # 'sample_images_in, sample_labels_in': Discriminator sample inputs
    # 'dsc_labels_in': Correct / incorrect labels for discriminator inputs (Fully-correct vs. mismatched tags and images)
    def __init__(self, args, noise_in, labels_in, sample_batch, mismatched_batch): #labels_in, sample_images_in, sample_labels_in, dsc_labels_in):
        self.args = args

        # Both sample_batch and mismatched_batch are tensor tuples of form:
        # (normalized_image, tags, smoothed_discriminator_labels)

        with tf.variable_scope('generator'):
            self.gen_out = self.generator(noise_in, labels_in)

            self.gen_img_out = gen_image_processing(self.gen_out)
            tf.summary.image('Generated Images', self.gen_img_out, collections=['gen-summaries'])

        with tf.variable_scope('discriminator'):
            self.dsc_fake_out = self.discriminator(self.gen_out, labels_in)

        with tf.variable_scope('discriminator', reuse=True):
            self.dsc_sample_out = self.discriminator(sample_batch[0], sample_batch[1])
            self.dsc_mismatch_out = self.discriminator(mismatched_batch[0], mismatched_batch[1])

        # Discriminator outputs probability that sample came from training dataset
        batch_size = tf.shape(self.dsc_fake_out)[0]
        self.dsc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dsc_fake_out,
            labels=tf.random_uniform([batch_size], minval=0.0, maxval=0.3),#tf.zeros_like(self.dsc_fake_out),
            name='discriminator-fake-loss'
        ))

        tf.summary.scalar('Discriminator Fake Loss', self.dsc_fake_loss, collections=['dsc-summaries'])

        self.dsc_sample_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dsc_sample_out,
            labels=sample_batch[2],
            name='discriminator-sample-loss'
        ))

        tf.summary.scalar('Discriminator Real Loss', self.dsc_sample_loss, collections=['dsc-summaries'])

        self.dsc_mismatch_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dsc_mismatch_out,
            labels=mismatched_batch[2],
            name='discriminator-mismatch-loss'
        ))

        tf.summary.scalar('Discriminator Mismatched Loss', self.dsc_mismatch_loss, collections=['dsc-summaries'])

        # NOTE: maybe also add image total variation to generator loss?
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dsc_fake_out,
            labels=tf.random_uniform([batch_size], minval=0.7, maxval=1.2),#tf.ones_like(self.dsc_fake_out),
            name='generator-loss'
        ))

        tf.summary.scalar('Generator Loss', self.gen_loss, collections=['gen-summaries'])

        self.dsc_loss = self.dsc_fake_loss + self.dsc_sample_loss + self.dsc_mismatch_loss

        tf.summary.scalar('Discriminator Loss', self.dsc_loss, collections=['dsc-summaries'])

        self.gen_vars = slim.get_variables(scope='generator')
        self.dsc_vars = slim.get_variables(scope='discriminator')

        dsc_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1)
        gen_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1)

        dsc_gv = dsc_optimizer.compute_gradients(self.dsc_loss, var_list=self.dsc_vars)
        gen_gv = gen_optimizer.compute_gradients(self.gen_loss, var_list=self.gen_vars)

        dsc_mean_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) for gv in dsc_gv])
        gen_mean_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) for gv in gen_gv])

        tf.summary.scalar('Mean Discriminator Gradients', dsc_mean_grads, collections=['dsc-summaries'])
        tf.summary.scalar('Mean Generator Gradients', gen_mean_grads, collections=['gen-summaries'])

        self.dsc_train = dsc_optimizer.apply_gradients(dsc_gv)
        self.gen_train = gen_optimizer.apply_gradients(gen_gv)

        #self.dsc_train = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(self.dsc_loss, var_list=self.dsc_vars)
        #self.gen_train = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(self.gen_loss, var_list=self.gen_vars)

        self.gen_summaries = tf.summary.merge_all(key='gen-summaries')
        self.dsc_summaries = tf.summary.merge_all(key='dsc-summaries')

    def gen_activation_fn(self):
        if self.args.gen_use_lrelu:
            return leaky_relu
        else:
            return tf.nn.relu

    # Z-vector: noise
    # Y-vector: tags / labels
    def generator(self, z, y):
        # Concat Z and Y, then project to DeConv stack input size
        gen_in = tf.concat([z, y], axis=1)

        proj_depth = self.args.gen_filter_base * (2 ** self.args.gen_layers)
        proj_height = conv_output_size(self.args.output_height, 2 ** (self.args.gen_layers + 1))
        proj_width = conv_output_size(self.args.output_width, 2 ** (self.args.gen_layers + 1))

        net = slim.fully_connected(gen_in, proj_depth * proj_height * proj_width, activation_fn=self.gen_activation_fn())
        net = tf.reshape(net, [-1, proj_height, proj_width, proj_depth])

        for layer_n in range(self.args.gen_layers):
            n_filters = self.args.gen_filter_base * (2 ** (self.args.gen_layers - layer_n - 1))
            net_h = conv_output_size(self.args.output_height, 2 ** (self.args.gen_layers - layer_n))
            net_w = conv_output_size(self.args.output_width, 2 ** (self.args.gen_layers - layer_n))

            net = slim.conv2d_transpose(net, n_filters,
                kernel_size=self.args.gen_kernel_size,
                stride=2,
                activation_fn=self.gen_activation_fn(),
                normalizer_fn=slim.fused_batch_norm
            )
            net = tf.reshape(net, [-1, net_h, net_w, n_filters])

        # Output layer:
        net = slim.conv2d_transpose(net, 3, kernel_size=self.args.gen_kernel_size,
            stride=2,
            activation_fn=tf.tanh
        )
        out = tf.reshape(net, [-1, self.args.output_height, self.args.output_width, 3])

        return out

    def discriminator(self, disc_in, labels):
        net = disc_in
        batch_size, in_h, in_w, in_d = tf.shape(disc_in)
        for layer_n in range(self.args.dsc_layers):
            net_h = conv_output_size(in_h, 2 ** (layer_n + 1))
            net_w = conv_output_size(in_w, 2 ** (layer_n + 1))
            n_filters = self.args.dsc_filter_base * (2 ** layer_n)

            net = slim.conv2d(net, n_filters,
                kernel_size=self.args.dsc_kernel_size,
                stride=2,
                activation_fn=leaky_relu,
                normalizer_fn=slim.fused_batch_norm
            )

            net = tf.reshape(net, [-1, net_h, net_w, n_filters])

        _, label_vector_sz = tf.shape(labels)
        _, net_h, net_w, net_d = tf.shape(net)

        net_area = net_h * net_w
        pad_sz = net_area - label_vector_sz # Pad label vector to network area size

        padding = tf.zeros([batch_size, pad_sz], dtype=tf.float32)
        labels = tf.concat([labels, padding], axis=1)
        labels = tf.reshape(labels, [batch_size, net_h, net_w, 1])

        net = tf.concat([net, labels], axis=3)

        # Alternatively, we can just feed it straight into the FC layer
        # instead of going through this final 1x1 conv
        n_filters = self.args.dsc_filter_base * (2 ** self.args.dsc_layers)
        net = slim.conv2d(net,
            n_filters,
            kernel_size=1,
            activation_fn=leaky_relu,
            normalizer_fn=slim.fused_batch_norm
        )

        net = tf.reshape(net, [-1, net_h, net_w, n_filters])

        net = tf.reshape(net, [batch_size, -1])
        out = slim.fully_connected(
            net,
            1,
            activation_fn=None #activation_fn=tf.sigmoid
        )

        return out
