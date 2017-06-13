import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.framework as framework
import numpy as np

import configargparse
import os
import sys
import datetime

from fnst import vgg, losses, image_transform_net

# This file implements operations necessary to train the feedforward CNN for Fast Style Transfer.

def vgg_preprocessing(input_img_data):
    ret = tf.to_float(input_img_data)
    return ret - tf.constant([123.68, 116.779, 103.939]) # Subtract mean pixel values

def decode_image_pipeline(input_filedata, input_filename):
    fnparts = os.path.splitext(input_filename)
    ext = fnparts[1]
    if ext == '.png':
        return tf.image.decode_png(input_filedata, channels=3)
    elif ext == '.jpg' or ext == '.jpeg':
        return tf.image.decode_jpeg(input_filedata, channels=3)

# Get layer Tensors from the VGG-19 output collection dictionary
# Additionally, squeeze them all (to remove the batch dimension)
def get_style_layers(output_dict, scope_name='vgg_19'):
    #print(str(output_dict))
    return [
        output_dict[scope_name+'/conv1/conv1_1'],
        output_dict[scope_name+'/conv2/conv2_1'],
        output_dict[scope_name+'/conv3/conv3_1'],
        output_dict[scope_name+'/conv4/conv4_1'],
        output_dict[scope_name+'/conv5/conv5_1'],
    ]

# Ditto for the content representation layer
def get_content_layer(output_dict, scope_name='vgg_19'):
    return output_dict[scope_name+'/conv4/conv4_2']

# Constructs the VGG network graph with the appropriate settings
def get_vgg_layers(input_tensor, args, reuse=True):
    vgg_device = None
    if args.distributed:
        vgg_device = '/job:worker/task:{:d}'.format(args.task_index)

    with tf.device(vgg_device): # Always keep VGG pinned locally
        #with slim.arg_scope([slim.variable, slim.model_variable], device='/cpu:0'):
        with slim.arg_scope([slim.conv2d], reuse=reuse): # Reuse VGG weights with precompute and training compute
            _, layers = vgg.vgg_19(input_tensor, is_training=False, spatial_squeeze=False)
            return layers

def init_vgg_model(args, session, vgg_saver):
    print('Loading VGG model weights from {}...'.format(args.vgg_19_checkpoint))
    sys.stdout.flush()

    vgg_saver.restore(session, args.vgg_19_checkpoint)

    print('VGG model weights loaded!')
    sys.stdout.flush()

# Computes the Gram matrices for the constant style images.
# 'filenames' should be a list of paths to the style images.
def precompute_gram_matrices(args, filenames, session_target=''):
    print('Building precompute graph...')
    sys.stdout.flush()

    image_tensors = []
    for filename in filenames:
        print('Loading file: {}'.format(filename))
        handle = open(filename, 'rb')
        data = handle.read()
        handle.close()

        fd_tensor = tf.constant(data, dtype=tf.string, name='data-'+os.path.basename(filename))
        decoded_tensor = decode_image_pipeline(fd_tensor, filename)
        model_input = vgg_preprocessing(decoded_tensor)
        model_input = tf.image.resize_bicubic([model_input], args.image_size)
        model_input = tf.squeeze(model_input) #tf.reshape(model_input, [args.image_size[0], args.image_size[1], 3])

        image_tensors.append(model_input)

    stacked_images = tf.parallel_stack(image_tensors)
    activations = get_vgg_layers(stacked_images, args, reuse=False)
    #_, activations = vgg.vgg_19(stacked_images, is_training=False, spatial_squeeze=False)

    style_layers = get_style_layers(activations)
    n_style_images = len(filenames)

    image_gram_matrices = []

    for layer_n, batched_layer in enumerate(style_layers):
        stacked_gram_matrices = losses.batched_gram_matx(args, batched_layer)
        image_gram_matrices.append(stacked_gram_matrices)

    vgg_model_vars = slim.get_model_variables(scope='vgg_19')
    precompute_vgg_saver = tf.train.Saver(vgg_model_vars)

    print("Launching precompute graph...")
    sys.stdout.flush()

    # Even in a distributed environment, we can just run this locally
    # -- it's not as expensive as running training.
    with tf.Session() as stage1:
        print('initializing global variables...')
        sys.stdout.flush()

        stage1.run(tf.global_variables_initializer())
        init_vgg_model(args, stage1, precompute_vgg_saver)
        #precompute_vgg_saver.restore(stage1, args.vgg_19_checkpoint)

        print('Precomputing style image activations...')
        sys.stdout.flush()

        actual_gram_matrices = stage1.run(image_gram_matrices)

    return actual_gram_matrices, precompute_vgg_saver

# Builds the model network and other ops for training.
# Inputs:
# - args: a configargparse Namespace object.
# - content_input: a Tensor containing batched input images (4D, shape must be known at construction time)
# - style_gram_matrices: Precomputed style Gram matrices (returned from the above function)
# - style_image_weights: Per-file weights for each style image.
# Outputs:
# - transformed_content: The image transform network to use for training.
# - total_loss: The overall loss to use for training.
def build_training_network(args, content_input, style_gram_matrices, style_image_weights):
    print("Building main compute graph...")
    sys.stdout.flush()

    transformed_content = image_transform_net.transform_net(content_input, args)

    tf.summary.image('Transform Network Output', transformed_content, max_outputs=1)

    # Add ops to compute losses:

    # First run everything through VGG-19:
    batch_sz = content_input.shape.as_list()[0]
    vgg_input = tf.concat([transformed_content, content_input], axis=0)
    vgg_input = vgg_preprocessing(vgg_input)

    print("VGG input shape: {}".format(vgg_input.shape))
    sys.stdout.flush()

    # Reuse VGG model weights from precompute
    vgg_layers = get_vgg_layers(vgg_input, args, reuse=True)
    #_, vgg_layers = vgg.vgg_19(vgg_input, is_training=False, spatial_squeeze=False)

    # Now get the layers of interest (preserving the batch dimension)
    style_layers = get_style_layers(vgg_layers, scope_name='vgg_19_1')
    content_layer = get_content_layer(vgg_layers, scope_name='vgg_19_1')

    # Compute content losses:
    transformed_content_layers = tf.slice(content_layer, [0,0,0,0],[batch_sz,-1,-1,-1])
    original_content_layers = tf.slice(content_layer, [batch_sz,0,0,0],[-1,-1,-1,-1])

    batched_content_loss = losses.content_loss(original_content_layers, transformed_content_layers)
    #batched_content_loss *= args.content_loss_weight

    # Subcomponents of overall style loss
    style_loss_components = []

    # Compute style loss subcomponents for each transformed image, style image, and actual layer
    for layer_idx, vgg_layer in enumerate(style_layers):
        transformed_input_layer = tf.slice(vgg_layer, [0, 0, 0, 0], [batch_sz, -1, -1, -1])
        style_losses = losses.batched_layer_style_loss(args, transformed_input_layer, style_gram_matrices[layer_idx])

        style_loss_components.append(style_losses)

    content_loss = tf.reduce_sum(batched_content_loss)
    variation_loss = tf.reduce_sum(tf.image.total_variation(transformed_content)) * args.variation_loss_weight
    style_loss = tf.reduce_sum(style_loss_components)

    total_loss = content_loss + variation_loss + style_loss

    tf.summary.scalar('Variation Loss', variation_loss)
    tf.summary.scalar('Content Loss', content_loss)
    tf.summary.scalar('Style Loss', style_loss)
    tf.summary.scalar('Total Loss', total_loss)

    return transformed_content, total_loss

# Builds ops for the input pipeline:
def build_input_ops(args):
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(args.content_image, name='input-filenames'), name='filename-producer')
    imreader = tf.WholeFileReader(name='image-reader')

    filename, filedata = imreader.read(filename_queue)
    #filedata = tf.Print(filedata, [filename], 'Processing as content: ')

    imdata = tf.image.decode_image(filedata, channels=3)
    imdata = tf.image.convert_image_dtype(imdata, tf.float32)

    # Enforce image size constraints (also catch stray GIFs)
    imdata = tf.image.resize_bicubic(tf.expand_dims(imdata, 0), args.image_size)
    imdata = tf.reshape(imdata, args.image_size+[3])

    training_batch = tf.train.shuffle_batch(
        [imdata],
        batch_size=args.batch_size,
        capacity=args.input_queue_capacity,
        min_after_dequeue=args.batch_size*4,
        num_threads=args.input_threads,
        shared_name='training-input-queue'
    )

    #print("Batch output Op shape: {}".format(str(training_batch.shape)))
    #sys.stdout.flush()

    return training_batch

# Builds ops for optimization, checkpointing, and other miscellaneous things.
def build_auxillary_ops(args, is_chief, loss):
    # Model Save/Load:
    transform_model_vars = image_transform_net.network_parameters(args)
    transform_model_saver = tf.train.Saver(transform_model_vars)

    # Optimization ops
    global_step = framework.get_or_create_global_step()

    optimizer = tf.train.AdamOptimizer(
        learning_rate=args.learning_rate,
        epsilon=args.adam_epsilon
    )

    train_step = optimizer.minimize(loss, global_step, transform_model_vars)

    scaffold = tf.train.Scaffold(saver=transform_model_saver)

    summary_writer = tf.summary.FileWriter(args.logdir + '-{}-{}'.format(args.job, args.task_index), tf.get_default_graph(), flush_secs=15)

    hooks = [
        tf.train.NanTensorHook(loss),
        tf.train.SummarySaverHook(save_secs=args.summary_save_frequency, scaffold=scaffold, summary_writer=summary_writer),
    ]


    chief_hooks=[
        tf.train.CheckpointSaverHook(args.checkpoint_dir, save_secs=600, scaffold=scaffold),
        tf.train.StepCounterHook(summary_writer=summary_writer),
    ]

    return train_step, scaffold, hooks, chief_hooks, summary_writer

# Runs precompute and builds the model + aux. ops.
# Can be run in a tf.Device context if necessary.
def setup_training(args, is_chief=True, server=None, cluster=None):
    session_target=''
    if server is not None:
        session_target = server.target

    style_image_paths = []
    style_image_weights = []
    for style_img_spec in args.style_image:
        p = style_img_spec.split(':')

        style_image_paths.append(p[0])
        style_image_weights.append(float(p[1]))

    input_device = None         # Device for input ops
    precompute_device = None    # Device for Gram matrix precomputation
    compute_device = None       # Device for main network computations
    if args.distributed:
        # Input processing is always pinned to the chief worker.
        # Each node's precomputations are always pinned to that node--
        #  this should prevent conflicts when restoring the VGG model.
        # Compute is distributed over each server using the typical Parameter Server / Worker model.
        input_device = '/job:worker/task:0'
        precompute_device = '/job:worker/task:{:d}'.format(args.task_index)
        compute_device = tf.train.replica_device_setter(
            cluster=cluster,
            worker_device='/job:worker/task:{:d}'.format(args.task_index)
        )

    with tf.device(precompute_device):
        gram_matrices, vgg_saver = precompute_gram_matrices(args, style_image_paths, session_target)

    with tf.device(input_device):
        train_input = build_input_ops(args)

    with tf.device(compute_device):
        transform_out, loss = build_training_network(args, train_input, gram_matrices, style_image_weights)
        train_step, scaffold, hooks, chief_hooks, summary_writer = build_auxillary_ops(args, is_chief, loss)

    return train_step, scaffold, hooks, chief_hooks, summary_writer, vgg_saver

# Adds training-specific parameters to an ArgumentParser.
def add_training_args(parser):
    parser.add_argument('--training', action='store_true', help='Perform transform model training.')

    parser.add_argument('--learning-rate', type=float, default=0.001, help='Transformation network learning rate.')
    parser.add_argument('--adam-epsilon', type=float, default=1e-08, help='Epsilon value for Adam optimizer.')

    parser.add_argument('--content-loss-weight', default=8e-4, type=float, help='Alpha parameter for loss calculation.')
    parser.add_argument('--variation-loss-weight', default=1e-5, type=float, help='Weighting factor for total variation in losses.')

    parser.add_argument('--image-size', required=True, type=int, nargs=2, help='Height and width of ALL images used for training (both style and content)')
    parser.add_argument('--content-image', required=True, action='append', help='File pattern (glob) matching training input (content) images. All inputs must be either PNGs or JPEGs.')
    parser.add_argument('--style-image', required=True, action='append', help='An input style image path:weight pair (i.e. my_image.jpg:1.0).')
    parser.add_argument('--vgg-19-checkpoint', default='vgg_19.ckpt', help='Checkpoint file containing VGG-19 model weights')

    parser.add_argument('--session-trace-frequency', type=int, default=20, help='Trace and output session information every N training steps.')
    parser.add_argument('--console-output-frequency', type=int, default=20, help='Print step information every N training steps.')
    parser.add_argument('--summary-save-frequency', type=int, default=25, help='Save summaries every N seconds.')
    parser.add_argument('--logdir', default='nst-logdir', help='Directory to save summaries to')

    parser.add_argument('--input-queue-capacity', type=int, default=10000, help='Maximum number of images to keep prefetched in the input image queue.')
    parser.add_argument('--batch-size', type=int, default=10, help='Training batch size.')
    parser.add_argument('--training-epochs', type=int, default=None, help='Number of training epochs to perform')
    parser.add_argument('--input-threads', type=int, default=4, help='Number of threads for input prefetching.')
