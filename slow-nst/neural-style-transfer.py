import tensorflow as tf
from slim.nets import vgg
import tensorflow.contrib.slim as slim
import numpy as np
import scipy as sp
import scipy.misc

import configargparse
import os
import sys
import datetime

# Calculates the feature correlation matrix for a 3D tensor
# (no batch dimension)
def feat_gram_matx_3d(layer, name='feat_gram_matx_3d'):
    return tf.tensordot(layer, layer, axes=[[0,1], [0,1]], name=name)

# Computes layer style losses given two feature maps from two layers
# Note that both layers should be 3D tensors (i.e. no batch dimensions)
def layer_style_loss(original_layer, generated_layer, name='layer_style_loss'):
    a = feat_gram_matx_3d(original_layer, name=name+'-original-tdot')
    g = feat_gram_matx_3d(generated_layer, name=name+'-generated-tdot')

    base_shape = tf.shape(original_layer) # should equal tf.shape(generated_layer) too
    m = tf.to_float(base_shape[0] * base_shape[1])
    n = tf.to_float(base_shape[2])

    e_unscaled = tf.reduce_sum(tf.square(g - a))

    return tf.reciprocal(4.0 * tf.square(n) * tf.square(m)) * e_unscaled

# Computes total style losses given a list of layer activation Tensors for each image
# (still no batch dimensions for the tensors)
def style_loss(original_layers, generated_layers, loss_weighting, name='style_loss'):
    layer_losses = []
    for a_layer, x_layer, w, i in zip(original_layers, generated_layers, loss_weighting, range(len(original_layers))):
        err = layer_style_loss(a_layer, x_layer, name=name+'-L{}'.format(i))
        layer_losses.append(w * err)

    return tf.reduce_sum(layer_losses, name=name+'-sum-weighted')

# Extracts the content loss by computing squared-error loss between activations
# on one layer
def content_loss(original_layer, generated_layer):
    return tf.reduce_sum(tf.square(generated_layer - original_layer) / 2.0)

# Get layer Tensors from the VGG-19 output collection dictionary
# Additionally, squeeze them all (to remove the batch dimension)
def get_style_layers(output_dict, scope_name='vgg_19'):
    #print(str(output_dict))
    return [
        tf.squeeze(output_dict[scope_name+'/conv1/conv1_1']),
        tf.squeeze(output_dict[scope_name+'/conv2/conv2_1']),
        tf.squeeze(output_dict[scope_name+'/conv3/conv3_1']),
        tf.squeeze(output_dict[scope_name+'/conv4/conv4_1']),
        tf.squeeze(output_dict[scope_name+'/conv5/conv5_1']),
    ]

# Ditto for the content representation layer
def get_content_layer(output_dict, scope_name='vgg_19'):
    return tf.squeeze(output_dict[scope_name+'/conv4/conv4_2'])

def model_preprocessing(input_img_data):
    ret = tf.to_float(input_img_data)
    return ret - tf.constant([123.68, 116.779, 103.939]) # Subtract mean pixel values

def decode_image_pipeline(input_filedata, input_filename):
    fnparts = os.path.splitext(input_filename)
    ext = fnparts[1]
    if ext == '.png':
        return tf.image.decode_png(input_filedata, channels=3)
    elif ext == '.jpg' or ext == '.jpeg':
        return tf.image.decode_jpeg(input_filedata, channels=3)

# Performs precomputation for content and style images.
def build_graph_stage1(args):
    print('Building precompute graph...')
    sys.stdout.flush()

    file_contents = {}

    filenames = [args.content_image]
    file_weights = [] # don't include content weighting
    for fp in args.style_image:
        pair = fp.split(':')
        filenames.append(pair[0])
        file_weights.append(float(pair[1]))

    image_tensors = []
    for filename in filenames:
        handle = open(filename, 'rb')
        data = handle.read()
        handle.close()

        fd_tensor = tf.constant(data, dtype=tf.string, name='data-'+os.path.basename(filename))
        decoded_tensor = decode_image_pipeline(fd_tensor, filename)
        model_input = model_preprocessing(decoded_tensor)
        model_input = tf.reshape(model_input, [args.image_size[0], args.image_size[1], 3])

        image_tensors.append(model_input)

    stacked_images = tf.parallel_stack(image_tensors)

    _, activations = vgg.vgg_19(stacked_images, is_training=False, spatial_squeeze=False)

    content_layer = get_content_layer(activations)[0]   # 1st batch element == content image
    style_layers = get_style_layers(activations)        # All other batch elements == style images

    n_style_images = len(filenames)-1
    style_activations = [[] for i in range(n_style_images)] # 2D list of activations for each style image

    for batched_layer in style_layers:
        unbatched = tf.unstack(batched_layer, num=len(filenames))[1:]
        for j, activation in enumerate(unbatched):
            style_activations[j].append(activation)

    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    saver = tf.train.Saver(model_vars)

    print("Launching precompute graph...")
    sys.stdout.flush()

    with tf.Session() as stage1:
        print('Loading model weights from {} and initializing global variables...'.format(args.vgg_19_checkpoint))
        sys.stdout.flush()

        stage1.run(tf.global_variables_initializer())
        saver.restore(stage1, args.vgg_19_checkpoint)

        print('Precomputing content and style image activations...')
        sys.stdout.flush()

        fetches = stage1.run([content_layer, style_activations])

        actual_content_layer = fetches[0]
        actual_style_activations = fetches[1]

    return actual_content_layer, actual_style_activations, file_weights

# Builds the portions of the graph that actually carry out the main computation
def build_graph_stage2(args, content_layer, style_activations, file_weights):
    generated_image_data = tf.placeholder(tf.float32, shape=args.image_size[0]*args.image_size[1]*3, name='generated_image')
    generated_image = tf.reshape(generated_image_data, [1]+args.image_size+[3])

    img_summary = tf.summary.image('Generated Image', generated_image, collections=[])

    generated_image = model_preprocessing(generated_image)

    # Run the generated image through the ConvNet model...
    _, generated_outputs = vgg.vgg_19(generated_image, is_training=False, spatial_squeeze=False)

    # Extract important layers from the model collection dictionary.
    # Since we build stage2 in its own graph, we use the default scope name for finding the layers.
    generated_content_layer = get_content_layer(generated_outputs, 'vgg_19')
    generated_style_layers = get_style_layers(generated_outputs, 'vgg_19')

    # Convert the precomputed layers to constant tensors
    constant_content_layer = tf.squeeze(tf.constant(content_layer, dtype=tf.float32, name='constant_content_input'))
    constant_style_layers = []
    for i, style_image in enumerate(style_activations):
        t = []
        for j, layer in enumerate(style_image):
            t.append(
                tf.squeeze(tf.constant(layer, dtype=tf.float32, name='constant_style_activation-{}-{}'.format(i,j)))
            )
        constant_style_layers.append(t)

    # Calculate loss subcomponents w/ weights
    wcl = args.content_loss_weight * content_loss(constant_content_layer, generated_content_layer)
    tf.summary.scalar('Weighted Content Loss', wcl)

    weighted_losses = [wcl]
    for i, image_layers in enumerate(constant_style_layers):
        image_style_loss = style_loss(image_layers, generated_style_layers, [0.2] * 5, name='style-loss-I{}'.format(i))
        weighted_loss = image_style_loss * file_weights[i]

        tf.summary.scalar('Weighted Style Loss {}'.format(i), weighted_loss)
        weighted_losses.append(weighted_loss)

    # Compute overall loss
    total_loss = tf.reduce_sum(weighted_losses)
    tf.summary.scalar('Total Loss', total_loss)

    # Compute gradients
    gen_grad = tf.gradients(total_loss, generated_image)
    flattened_gradients = tf.reshape(gen_grad, [-1])
    tf.summary.scalar('Mean Gradient', tf.reduce_mean(gen_grad))

    # Auxillary stuff
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    saver = tf.train.Saver(model_vars)

    increment_step = tf.assign_add(tf.contrib.framework.get_or_create_global_step(), 1)

    outputs = {
        'loss': total_loss,
        'grad': flattened_gradients,
        'summary': tf.summary.merge_all(),
        'image_summary': img_summary,
        'increment_step': increment_step,
    }

    return generated_image_data, outputs, saver

def init_compute(args, cluster=None, server=None):
    precompute_graph = tf.Graph()

    dev_fn = None
    session_target = None
    if args.distributed:
        dev_fn = tf.train.replica_device_setter(
            worker_device="/job:worker/task:{:d}".format(args.task_index),
            cluster=cluster
        )
        session_target = server.target

    with slim.arg_scope([slim.variable, slim.model_variable], device='/cpu:0'):
        with precompute_graph.as_default():
            content_layer, style_acts, style_weights = build_graph_stage1(args)

    print('Building main compute graph.')
    sys.stdout.flush()

    with slim.arg_scope([slim.variable, slim.model_variable], device='/cpu:0'):
        with tf.device(dev_fn):
            image_input_tensor, outputs, model_saver = build_graph_stage2(args, content_layer, style_acts, style_weights)

    #tf.train.write_graph(precompute_graph, args.logdir, 'precompute-graph.pbtxt')
    #tf.train.write_graph(tf.get_default_graph(), args.logdir, 'compute-graph.pbtxt')

    print("Launching main graph...")
    sys.stdout.flush()

    sess = tf.Session(target=session_target)

    summ_writer = tf.summary.FileWriter(args.logdir, graph=sess.graph)

    print("Initializing variables...")
    sys.stdout.flush()
    sess.run(tf.global_variables_initializer())

    print('Reloading model weights from {}...'.format(args.vgg_19_checkpoint))
    sys.stdout.flush()
    model_saver.restore(sess, args.vgg_19_checkpoint)

    # Prepare initial image and bounds for L-BFGS-B
    img = None
    if args.continue_optimization is not None:
        img = scipy.misc.imread(args.continue_optimization).flatten()
        print("Loaded interim image! Continuing...")
        sys.stdout.flush()
    else:
        img = np.random.uniform(0.0, 255.0, size=args.image_size[0]*args.image_size[1]*3)
    bounds = np.zeros((args.image_size[0]*args.image_size[1]*3, 2))
    bounds[:] = (0.0, 255.0)

    # Get our loss function for L-BFGS-B:
    current_iteration = 1

    ###
    def get_loss(generated_img):
        nonlocal current_iteration

        if current_iteration % args.iteration_output_freq == 0:
            print("[{}] - beginning iteration {}".format(str(datetime.datetime.now()), current_iteration))
            sys.stdout.flush()

        loss, grads, summ, step = sess.run(
            (outputs['loss'], outputs['grad'], outputs['summary'], outputs['increment_step']),
            feed_dict={ image_input_tensor: generated_img }
        )

        current_iteration = step+1
        summ_writer.add_summary(summ, step)

        if args.interim_file_template is not None and args.interim_save_freq is not None:
            if step % args.interim_save_freq == 0:
                interm_img = np.reshape(generated_img, args.image_size + [3])
                scipy.misc.imsave(args.interim_file_template + '{}.png'.format(step), interm_img)

                img_summ = outputs['image_summary'].eval(session=sess, feed_dict={ image_input_tensor: generated_img })
                summ_writer.add_summary(img_summ, step)

        if step % args.iteration_output_freq == 0:
            print('[{}] Iteration {} - loss: {}'.format(str(datetime.datetime.now()), step, loss))
            sys.stdout.flush()

        return np.float64(loss), np.float64(grads) # SciPy doesn't like it when you pass float32's to minimize...
    ###

    return sess, img, bounds, get_loss

def main_compute_loop(args):
    ps_hosts = None
    cluster = None
    server = None

    if not os.path.exists(args.logdir):
        if os.path.exists(os.path.dirname(args.logdir)):
            os.makedirs(args.logdir)
        else:
            raise FileNotFoundError("Logging base directory {} does not exist.".format(os.path.dirname(args.logdir)))

    if args.interim_file_template is not None:
        interim_img_dir = os.path.dirname(args.interim_file_template)
        if not os.path.exists(interim_img_dir):
            if os.path.exists(os.path.dirname(interim_img_dir)):
                os.makedirs(interim_img_dir)
            else:
                raise FileNotFoundError("Intermediate image base directory {} does not exist.".format(os.path.dirname(interim_img_dir)))

    if args.distributed:
        print("Initializing distributed mode.")
        sys.stdout.flush()

        ps_hosts = args.ps_hosts.split(',')
        cluster = tf.train.ClusterSpec({
            "ps": ps_hosts,
            "worker": [args.worker_host],
        })
        server = tf.train.Server(cluster, job_name=args.job, task_index=args.task_index)

    if args.distributed and args.job == 'ps':
        print("Starting parameter server.")
        sys.stdout.flush()
        server.join()
    else:
        sess, init_img, bounds, loss_fn = init_compute(args, cluster, server)

        print("Beginning image generation.")
        sys.stdout.flush()

        final_img, final_loss, info_dict = sp.optimize.fmin_l_bfgs_b(
            loss_fn,
            init_img,
            bounds=bounds,
            iprint=args.iteration_output_freq,
            maxiter=args.max_iterations,
        )

        print("Saving output...")
        sys.stdout.flush()

        output_img = np.reshape(final_img, args.image_size + [3])
        scipy.misc.imsave(args.outfile, output_img)

# Main script code below:
parser = configargparse.ArgumentParser(description='Performs neural style transfer.', default_config_files=['~/.nst-settings'])
parser.add_argument('-c', '--config', is_config_file=True, help='Configuration file path')

parser.add_argument('--image_size', required=True, type=int, nargs=2, help='Height and width of images used for content, style, and output (i.e. --image_size 512 512)')
parser.add_argument('--content_image', required=True, help='Input content image')
parser.add_argument('--outfile', required=True, help='File to write completed image to (image dimensions will be determined from content image)')

parser.add_argument('--style_image', action='append', help='An input style image path:weight pair (i.e. my_image.jpg:1.0). This option can be specified multiple times.')
parser.add_argument('--content_loss_weight', default=8e-4, type=float, help='Alpha parameter for loss calculation')

parser.add_argument('--distributed', action='store_true', help='Run in distributed mode')
parser.add_argument('--ps_hosts', help='List of parameter server hostname:port pairs, comma-separated.')
parser.add_argument('--worker_host', help='The worker node\'s hostname:port pair')
parser.add_argument('--job', help='Job name for this instance.')
parser.add_argument('--task_index', type=int, default=0, help='Task index for specified job.')

parser.add_argument('--continue_optimization', default=None, help='Path to an intermediate image to resume from (begin from random data if not set)')
parser.add_argument('--vgg_19_checkpoint', default='vgg_19.ckpt', help='Checkpoint file containing VGG-19 model weights')
parser.add_argument('--logdir', default='nst-logdir', help='Directory to save summaries to')
parser.add_argument('--interim_file_template', default='nst-intermediate-', help='Intermediate file template (iteration number and .png extension will be appended)')
parser.add_argument('--interim_save_freq', default=10, type=int, help='How often to save intermediate images during optimization')
parser.add_argument('--iteration_output_freq', default=1, type=int, help='How often to print information to console during optimization')
parser.add_argument('--max_iterations', default=15000, type=int, help='Maximum iterations to run for output')

args = parser.parse_args()
main_compute_loop(args)
