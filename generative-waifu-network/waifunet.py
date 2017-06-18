import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import sys

def debug_print(line):
    print(line)
    sys.stdout.flush()

def conv_output_size(dim_in, stride):
    if isinstance(dim_in, tf.Dimension):
        if dim_in.value is None:
            raise ValueError("Dimension size unknown!")
        dim_in = dim_in.value
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
    parser.add_argument('--z-size', type=int, default=256, help='Dimensionality of Z (noise) vectors')
    parser.add_argument('--label-size', type=int, default=1000, help='Dimensionality of Y (tag) vectors')
    parser.add_argument('--output-height', type=int, default=1024, help='Height of output images')
    parser.add_argument('--output-width', type=int, default=1024, help='Width of output images')
    parser.add_argument('--gen-filter-base', type=int, default=64, help='Final (base) number of generator deconv filters (before image output layer)')
    parser.add_argument('--gen-layers', type=int, default=4, help='Number of generator deconv layers (not including output layer)')
    parser.add_argument('--gen-kernel-size', type=int, default=5, help='Height+Width of generator deconv layer kernels')
    parser.add_argument('--dsc-kernel-size', type=int, default=5, help='Height+Width of discriminator conv layer kernels')
    parser.add_argument('--dsc-filter-base', type=int, default=64, help='Initial number of input filters for discriminator network')
    parser.add_argument('--dsc-layers', type=int, default=4, help='Number of conv layers in discriminator network (not including output flattening + sigmoid FC output layer)')

    # Alternately: 5e-5 for Wasserstein GANs
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate for generator and discriminator networks')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 parameter for Adam optimizers (for both generator and discriminator)')

    parser.add_argument('--generator-activation', default='lrelu', help='Activation function to use within the generator network (valid values are \'relu\', \'lrelu\', \'selu\')')
    parser.add_argument('--discriminator-activation', default='lrelu', help='Activation function to use within the discrimnator network (same values as --generator-activation)')

    parser.add_argument('--wasserstein', action='store_true', help='Use Wasserstein distance for optimization')
    parser.add_argument('--dsc-weight-clip', type=float, default=1e-2, help='Critic network weight clipping values (for Wasserstein GANs)')

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
        with slim.arg_scope([slim.variable, slim.model_variable], device='/cpu:0'):
            with tf.variable_scope('generator'):
                debug_print("[WaifuNet] Creating generator...")
                self.gen_out = self.generator(noise_in, labels_in)

                self.gen_img_out = gen_image_processing(self.gen_out)
                tf.summary.image('Generated Images', self.gen_img_out, collections=['gen-summaries'])

            with tf.variable_scope('discriminator'):
                debug_print("[WaifuNet] Creating discriminator (generated images)...")
                self.dsc_fake_out = self.discriminator(self.gen_out, labels_in)

                debug_print("[WaifuNet] Creating discriminator (sampled images)...")
                self.dsc_sample_out = self.discriminator(sample_batch[0], sample_batch[1], reuse=True)

            if not args.wasserstein:
                debug_print("[WaifuNet] Creating discriminator (mismatched images)...")

                self.dsc_mismatch_out = self.discriminator(mismatched_batch[0], mismatched_batch[1], reuse=True)

            if args.wasserstein:
                self.wasserstein_gan_loss()
            else:
                self.standard_gan_loss(sample_batch, mismatched_batch)

        # self.dsc_loss now contains discriminator / critic network loss tensor
        # self.gen_loss now contains generator network loss tensor

        tf.summary.scalar('Generator Loss', self.gen_loss, collections=['gen-summaries'])
        tf.summary.scalar('Discriminator Loss', self.dsc_loss, collections=['dsc-summaries'])

        self.gen_vars = slim.get_trainable_variables(scope='generator')
        self.dsc_vars = slim.get_trainable_variables(scope='discriminator')

        if not args.wasserstein:
            debug_print("[WaifuNet] Creating standard optimizer (Adam) ops...")

            dsc_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1)
            gen_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1)
        else:
            debug_print("[WaifuNet] Creating Wasserstein optimizer (RMSProp) ops...")

            dsc_optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
            gen_optimizer = tf.train.RMSPropOptimizer(args.learning_rate)

        debug_print("[WaifuNet] Creating gradient compute / apply ops...")

        dsc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
        gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')

        with tf.control_dependencies(gen_update_ops):
            gen_gv = gen_optimizer.compute_gradients(self.gen_loss, var_list=self.gen_vars)

        with tf.control_dependencies(dsc_update_ops):
            dsc_gv = dsc_optimizer.compute_gradients(self.dsc_loss, var_list=self.dsc_vars)

        for grad, var in dsc_gv:
            if grad is None:
                print("Gradient for {} is None!".format(var.name))

        for grad, var in gen_gv:
            if grad is None:
                print("Gradient for {} is None!".format(var.name))

        dsc_mean_grads = tf.reduce_mean([tf.reduce_mean(grad) for grad, var in dsc_gv])
        gen_mean_grads = tf.reduce_mean([tf.reduce_mean(grad) for grad, var in gen_gv])

        tf.summary.scalar('Mean Discriminator Gradients', dsc_mean_grads, collections=['dsc-summaries'])
        tf.summary.scalar('Mean Generator Gradients', gen_mean_grads, collections=['gen-summaries'])

        global_step = tf.contrib.framework.get_or_create_global_step()

        dsc_apply_grads = dsc_optimizer.apply_gradients(dsc_gv, global_step=global_step)
        self.gen_train = gen_optimizer.apply_gradients(gen_gv, global_step=global_step)

        if args.wasserstein:
            debug_print("[WaifuNet] Creating Wasserstein critic network weight clipping ops...")

            # add ops to clip the critic/discriminator network weights-- this must happen AFTER weights are updated
            with tf.control_dependencies([dsc_apply_grads]):
                clipped_weights = [tf.clip_by_value(w, -args.dsc_weight_clip, args.dsc_weight_clip) for w in self.dsc_vars]
                weight_clip_op = tf.group(*clipped_weights)
                self.dsc_train = weight_clip_op
        else:
            self.dsc_train = dsc_apply_grads

        self.gen_summaries = tf.summary.merge_all(key='gen-summaries')
        self.dsc_summaries = tf.summary.merge_all(key='dsc-summaries')

    def wasserstein_gan_loss(self):
        debug_print("[WaifuNet] Creating Wasserstein GAN loss ops...")

        critic_real_mean = tf.reduce_mean(self.dsc_sample_out)
        critic_fake_mean = tf.reduce_mean(self.dsc_fake_out)

        self.dsc_loss = critic_real_mean - critic_fake_mean
        self.gen_loss = critic_fake_mean

    def standard_gan_loss(self, sample_batch, mismatched_batch):
        debug_print("[WaifuNet] Creating standard GAN loss ops...")

        # Discriminator outputs probability that sample came from training dataset
        batch_size = self.dsc_fake_out.shape.as_list()[0]
        self.dsc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dsc_fake_out,
            labels=tf.random_uniform([self.args.batch_size], minval=0.0, maxval=0.3),#tf.zeros_like(self.dsc_fake_out),
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
            labels=tf.random_uniform([self.args.batch_size], minval=0.7, maxval=1.2),#tf.ones_like(self.dsc_fake_out),
            name='generator-loss'
        ))

        self.dsc_loss = self.dsc_fake_loss + self.dsc_sample_loss + self.dsc_mismatch_loss

    def gen_activation_fn(self):
        if self.args.gen_use_lrelu:
            return leaky_relu
        else:
            return tf.nn.relu

    def deception_module(self, tensor_in, output_depth, bottleneck_depth, name='Deception', reuse=False):
        batch_size, layer_height, layer_width, layer_depth = tensor_in.shape.as_list()
        output_height, output_width = layer_height * 2, layer_width * 2
        head_depth = output_depth // 4

        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=self.gen_activation_fn(), reuse=reuse):
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

    # Z-vector: noise
    # Y-vector: tags / labels
    def generator(self, z, y, reuse=False):
        with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected], activation_fn=self.gen_activation_fn(), reuse=reuse):
            # Concat Z and Y, then project to DeConv stack input size
            gen_in = tf.concat([z, y], axis=1)

            proj_depth = self.args.gen_filter_base * (2 ** self.args.gen_layers)
            proj_height = conv_output_size(self.args.output_height, 2 ** (self.args.gen_layers + 1))
            proj_width = conv_output_size(self.args.output_width, 2 ** (self.args.gen_layers + 1))

            net = slim.fully_connected(gen_in, proj_depth * proj_height * proj_width, scope='input')
            net = tf.reshape(net, [-1, proj_height, proj_width, proj_depth])

            for layer_n in range(self.args.gen_layers):
                n_filters = self.args.gen_filter_base * (2 ** (self.args.gen_layers - layer_n - 1))
                net_h = conv_output_size(self.args.output_height, 2 ** (self.args.gen_layers - layer_n))
                net_w = conv_output_size(self.args.output_width, 2 ** (self.args.gen_layers - layer_n))

                net = slim.conv2d_transpose(net, n_filters,
                    kernel_size=self.args.gen_kernel_size,
                    stride=2,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training': True, 'fused': True},
                    scope='conv{:d}'.format(layer_n+1)
                )
                net = tf.reshape(net, [-1, net_h, net_w, n_filters])

            # Output layer:
            net = slim.conv2d_transpose(net, 3, kernel_size=self.args.gen_kernel_size,
                stride=2,
                activation_fn=tf.tanh,
                scope='output'
            )
            out = tf.reshape(net, [-1, self.args.output_height, self.args.output_width, 3])

            return out

    def discriminator(self, disc_in, labels, reuse=False):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=reuse, activation_fn=leaky_relu):
            net = disc_in
            batch_size, in_h, in_w, in_d = disc_in.shape.as_list()
            for layer_n in range(self.args.dsc_layers):
                net_h, net_w = conv_output_size(in_h, 2 ** (layer_n + 1)), conv_output_size(in_w, 2 ** (layer_n + 1))
                n_filters = self.args.dsc_filter_base * (2 ** layer_n)

                net = slim.conv2d(net, n_filters,
                    kernel_size=self.args.dsc_kernel_size,
                    stride=2,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training': True, 'fused': True},
                    scope='conv{:d}'.format(layer_n+1),
                )

                net = tf.reshape(net, [-1, net_h, net_w, n_filters])

            _, label_vector_sz = labels.shape.as_list()
            _, net_h, net_w, net_d = net.shape.as_list()

            net_area = net_h * net_w

            # Use an FC layer to project the labels into a feature map
            label_feat_map = slim.fully_connected(labels, net_area, scope='label_project')
            label_feat_map = tf.reshape(label_feat_map, [batch_size, net_h, net_w, 1])

            net = tf.concat([net, label_feat_map], axis=3)

            # Alternatively, we can just feed it straight into the FC layer
            # instead of going through this final 1x1 conv
            n_filters = self.args.dsc_filter_base * (2 ** self.args.dsc_layers)
            net = slim.conv2d(net,
                n_filters,
                kernel_size=1,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': True, 'fused': True},
                scope='conv{:d}'.format(self.args.dsc_layers+1),
            )

            net = tf.reshape(net, [batch_size, -1])
            activation_fn = tf.sigmoid
            if self.args.wasserstein:
                activation_fn = None

            out = slim.fully_connected(
                net,
                1,
                activation_fn=activation_fn,
                scope='output',
            )

            return out
