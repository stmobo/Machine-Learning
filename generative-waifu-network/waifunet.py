import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import sys

from generator_network import *
from discriminator_network import *
from common import *

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

    # Alternately: 5e-5 for Wasserstein GANs
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate for generator and discriminator networks')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 parameter for Adam optimizers (for both generator and discriminator)')
    parser.add_argument('--n-gpus', type=int, default=1, help='Number of GPUs to use for training.')

    return parser

class waifunet(object):
    # Builds the model.
    # Options:
    # 'z_size' [default 256]: Length / dimensionality of Z vector
    # 'label_size' [default 1000]: Length / dimensionality of Y vector
    # 'output_width', 'output_height' [default 1024 for both]: Dimension of generator output

    # 'learning_rate' [recommended value 2e-4]: Learning rate for Adam / RMSProp optimization
    # 'beta1' [recommended value 0.5]: Beta1 parameter for Adam optimization
    #
    # Input tensor parameters:
    # 'noise_in', 'labels_in': Generator network inputs (noise vectors and tags)
    # 'sample_images_in, sample_labels_in': Discriminator sample inputs
    # 'dsc_labels_in': Correct / incorrect labels for discriminator inputs (Fully-correct vs. mismatched tags and images)
    def __init__(self, args, noise_in, labels_in, sample_batch, mismatched_batch): #labels_in, sample_images_in, sample_labels_in, dsc_labels_in):
        self.args = args

        self.noise = noise_in
        self.labels = labels_in
        self.samples = sample_batch
        self.mismatched = mismatched_batch

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)

    def gen_training_step(self, sess, summaries=False, trace=False):
        if trace:
            run_meta = tf.RunMetadata()
            run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        else:
            run_meta = None
            run_opts = None

        if summaries:
            _, gen_summary = sess.run([self.gen_train, self.gen_summaries], options=run_opts, run_metadata=run_meta)
        else:
            gen_summary = None
            sess.run(self.gen_train, options=run_opts, run_metadata=run_meta)

        return gen_summary, run_meta

    def dsc_training_step(self, sess, summaries=False, trace=False):
        if trace:
            run_meta = tf.RunMetadata()
            run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        else:
            run_meta = None
            run_opts = None

        if summaries:
            _, dsc_summary = sess.run([self.dsc_train, self.dsc_summaries], options=run_opts, run_metadata=run_meta)
        else:
            dsc_summary = None
            sess.run(self.dsc_train, options=run_opts, run_metadata=run_meta)

        return dsc_summary, run_meta

    # Network outputs:
    # self.dsc_train: Performs a single discriminator network training step when evaluated.
    # self.gen_train: Performs a single generator network training steps when evaluated.
    # self.dsc_summaries: Returns a merged summary tensor for a discriminator training step.
    # self.gen_summaries: Returns a merged summary tensor for a generator training step; includes images.
    # self.gen_images: Returns the generated images from the generator network (for each tower)
    def build(self):
        if self.args.n_gpus <= 1:
            # Single-tower
            self.single_tower_gradients()
        else:
            self.multi_tower_gradients()

        self.training_ops()
        self.summary_ops()

    def multi_tower_gradients(self):
        gen_grads = []
        dsc_grads = []

        gen_losses = []
        dsc_losses = []

        self.gen_images = []

        with tf.variable_scope('WaifuNet'):
            for i in range(self.args.n_gpus):
                with tf.device("/gpu:{:d}".format(i)):
                    grads, losses, nets = self.per_tower_ops()

                    tf.get_variable_scope().reuse_variables()

                    gen_grads.append(grads['gen'])
                    dsc_grads.append(grads['dsc'])

                    gen_losses.append(losses['gen'])
                    dsc_losses.append(losses['dsc'])

                    gen_out = gen_image_processing(nets['gen'].out)
                    self.gen_images.append(gen_out)

        # Now average gradients across all towers:
        self.gen_grads = self.average_gradients(gen_grads)
        self.dsc_grads = self.average_gradients(dsc_grads)

        # And average losses across all towers
        self.gen_loss = tf.reduce_mean(gen_losses)
        self.dsc_loss = tf.reduce_mean(dsc_loss)

    def single_tower_gradients(self):
        with tf.variable_scope('WaifuNet'):
            grads, losses, nets = self.per_tower_ops()
            self.gen_grads = grads['gen']
            self.dsc_grads = grads['dsc']

            self.gen_loss = losses['gen']
            self.dsc_loss = losses['dsc']

            gen_out = gen_image_processing(nets['gen'].out)
            self.gen_images = [gen_out]

    def training_ops(self):
        global_step = tf.contrib.framework.get_or_create_global_step()

        gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='WaifuNet/Generator')
        dsc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='WaifuNet/Discriminator')

        with tf.control_dependencies(gen_update_ops):
            self.gen_train = self.optimizer.apply_gradients(self.gen_grads, global_step=global_step)

        with tf.control_dependencies(dsc_update_ops):
            dsc_apply_grads = self.optimizer.apply_gradients(self.dsc_grads, global_step=global_step)

        if self.args.wasserstein:
            with tf.control_dependencies([dsc_apply_grads]):
                dsc_vars = slim.get_trainable_variables(scope='WaifuNet/Discriminator')
                clipped_weights = [tf.clip_by_value(w, -args.dsc_weight_clip, args.dsc_weight_clip) for w in dsc_vars]
                self.dsc_train = tf.group(*clipped_weights)
        else:
            self.dsc_train = dsc_apply_grads

    def summary_ops(self):
        tf.summary.scalar('Mean Generator Loss', self.gen_loss, collections=['gen-summaries'])
        tf.summary.histogram('Generator Gradients', self.gen_grads, collections=['gen-summaries'])
        tf.summary.image('Generator Output', self.gen_images[0], collections=['gen-summaries'])

        tf.summary.scalar('Mean Discriminator Loss', self.dsc_loss, collections=['dsc-summaries'])
        tf.summary.histogram('Discriminator Gradients', self.dsc_grads, collections=['dsc-summaries'])

        self.gen_summaries = tf.summary.merge_all(key='gen-summaries')
        self.dsc_summaries = tf.summary.merge_all(key='dsc-summaries')

    def average_gradients(self, tower_grads):
        average_grads = []
        for gv in zip(*tower_grads):
            grads = []
            var = gv[0][1]

            for grad, _ in gv:
                grads.append(tf.expand_dims(grad, 0))

            avg_grad = tf.concat(grads, axis=0)
            avg_grad = tf.reduce_mean(grads, axis=0)

            average_grads.append( (avg_grad, var) )

        return average_grads

    def per_tower_ops(self):
        gen = Generator(args, noise_in, labels_in)
        fake_image = gen.build()

        dsc_fake_net = Discriminator(args, fake_image, self.labels)
        dsc_real_net = Discriminator(args, self.samples[0], self.samples[1])

        dsc_fake = dsc_fake_net.build()
        dsc_real = dsc_real_net.build(reuse=True)

        if not self.args.wasserstein:
            dsc_mismatch_net = Discriminator(args, self.mismatched[0], self.mismatched[1])
            dsc_mismatch = dsc_mismatch_net.build(reuse=True)

            gen_loss, dsc_loss = self.standard_gan_loss(dsc_fake, dsc_real, dsc_mismatch)
        else:
            dsc_mismatch_net = None
            gen_loss, dsc_loss = self.wasserstein_gan_loss(dsc_fake, dsc_real)

        gen_grads = self.optimizer.compute_gradients(gen_loss, var_list=gen.vars)
        dsc_grads = self.optimizer.compute_gradients(dsc_loss, var_list=dsc_fake_net.vars)

        nets   = {'gen': gen, 'dsc_fake': dsc_fake_net, 'dsc_real': dsc_real_net, 'dsc_mismatch': dsc_mismatch_net}
        grads  = {'dsc': dsc_loss, 'gen': gen_loss}
        losses = {'dsc': dsc_loss, 'gen': gen_loss}

        return grads, losses, nets

    def wasserstein_gan_loss(self, dsc_fake_out, dsc_real_out):
        debug_print("[WaifuNet] Creating Wasserstein GAN loss ops...")

        critic_real_mean = tf.reduce_mean(dsc_real_out)
        critic_fake_mean = tf.reduce_mean(dsc_fake_out)

        dsc_loss = critic_real_mean - critic_fake_mean
        gen_loss = critic_fake_mean

        return gen_loss, dsc_loss

    def standard_gan_loss(self, dsc_fake_out, dsc_real_out, dsc_mismatch_out):
        debug_print("[WaifuNet] Creating standard GAN loss ops...")

        # Discriminator outputs probability that sample came from training dataset
        batch_size = self.dsc_fake_out.shape.as_list()[0]
        dsc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dsc_fake_out,
            labels=tf.random_uniform([self.args.batch_size], minval=0.0, maxval=0.3),#tf.zeros_like(self.dsc_fake_out),
            name='discriminator-fake-loss'
        ))

        dsc_sample_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dsc_real_out,
            labels=self.samples[2],
            name='discriminator-sample-loss'
        ))

        dsc_mismatch_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dsc_mismatch_out,
            labels=self.mismatched[2],
            name='discriminator-mismatch-loss'
        ))

        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dsc_fake_out,
            labels=tf.random_uniform([self.args.batch_size], minval=0.7, maxval=1.2),#tf.ones_like(self.dsc_fake_out),
            name='generator-loss'
        ))

        dsc_loss = dsc_fake_loss + dsc_sample_loss + dsc_mismatch_loss

        return gen_loss, dsc_loss
