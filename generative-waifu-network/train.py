import tensorflow as tf
import numpy as np
import configargparse
import sys

import waifunet
import discriminator_network
import generator_network
from common import *


def training_parameters(parser):
    parser.add_argument('--batch-size', type=int, default=16, help='Image batch size to use for training (per GPU / tower)')
    parser.add_argument('--input-queue-capacity', type=int, default=1000, help='Number of images to prefetch for the input queue')
    parser.add_argument('--input-threads', type=int, default=4, help='Number of threads to use for input prefetching')
    parser.add_argument('--dsc-steps', type=int, default=1, help='Number of discriminator / critic training steps to run for every generator training step')

    parser.add_argument('--input-filenames', action='append', help='Globs matching input TFRecord files')
    parser.add_argument('--checkpoint-dir', help='Directory to write checkpoint files to')
    parser.add_argument('--log-dir', help='Directory to write log files to')

    parser.add_argument('--summary-frequency', type=int, default=20, help='How often (in generator training iterations) to output summary info')
    parser.add_argument('--trace-frequency', type=int, default=500, help='How often to trace session run info')

def sample_pipeline(args):
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(args.input_filenames, name='input-filenames'), name='filename-producer')
    imreader = tf.TFRecordReader(name='image-reader')

    _, serialized_example = imreader.read(filename_queue)
    example = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'tags': tf.FixedLenFeature(shape=[], dtype=tf.string),
        }
    )

    tags_8u = tf.decode_raw(example['tags'], tf.uint8)
    tags_8u = tf.reshape(tags_8u, [args.label_size])

    im_8u = tf.image.decode_image(example['image'], channels=3)
    im_f32 = tf.image.convert_image_dtype(im_8u, tf.float32)

    # Resize all images to match the generator's output dimensions
    img_resized = tf.image.resize_bicubic(tf.expand_dims(im_f32, axis=0), (args.output_height, args.output_width))
    img_resized = tf.reshape(img_resized, [args.output_height, args.output_width, 3])

    # Scale all image values from range [0,1] to range [-1, 1] (same as TanH)
    img_out = (img_resized * 2.0) - 1.0

    # Convert tags to float tensors
    tags_f32 = tf.to_float(tags_8u)

    # label smoothing
    dsc_label = tf.random_uniform([], minval=0.7, maxval=1.2)

    # Outputs MATCHED images / tags
    sample_images, sample_tags, sample_labels = tf.train.shuffle_batch(
        [img_out, tags_f32, dsc_label],
        batch_size=args.batch_size,
        capacity=args.input_queue_capacity,
        min_after_dequeue=args.batch_size*4,
        num_threads=args.input_threads,
        shared_name='matched-input-queue'
    )

    sample_batch = (sample_images, sample_tags, sample_labels)

    # Noise values are sampled from a (truncated) normal distribution
    # and must fall within the range [0, 1]
    noise_batch = tf.truncated_normal([args.batch_size, args.z_size], mean=0.5, stddev=0.25)

    return sample_batch, noise_batch

def training_step(args, sess, summary_writer, wnet, global_step):
    write_summary = (global_step % args.summary_frequency == 0)
    trace_sessions = (global_step % args.trace_frequency == 0)

    if write_summary:
        debug_print("Starting step {}...".format(global_step))

    for dsc_step in range(args.dsc_steps):
        dsc_summary, dsc_meta = wnet.dsc_training_step(sess, summaries=(write_summary and dsc_step == 0), trace=(trace_sessions and dsc_step == 0))

        if write_summary:
            summary_writer.add_summary(dsc_summary, global_step=global_step)

        if trace_sessions:
            summary_writer.add_run_metadata(dsc_meta, "step-{:d}-dsc".format(global_step))

    gen_summary, gen_meta = wnet.gen_training_step(sess, summaries=write_summary,  trace=trace_sessions)

    if write_summary:
        debug_print("Completed step {}!".format(global_step))

    if write_summary:
        summary_writer.add_summary(gen_summary, global_step=global_step)

    if trace_sessions:
        summary_writer.add_run_metadata(gen_meta, "step-{:d}-gen".format(global_step))
        summary_writer.flush()

def do_training(args):
    debug_print("Creating input pipeline...")
    sample_batch, noise_batch = sample_pipeline(args)

    debug_print("Creating main networks...")
    wnet = waifunet.waifunet(args, noise_batch, sample_batch[1], sample_batch)

    return # debugging only

    summ_writer = tf.summary.FileWriter(
        args.log_dir,
        graph=tf.get_default_graph()
    )

    debug_print("Launching graph...")
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=args.checkpoint_dir,
        save_summaries_steps=None,
    ) as mon_sess:
        debug_print("Created session!")
        step = 0
        while not mon_sess.should_stop():
            training_step(args, mon_sess, summ_writer, wnet, step)
            step += 1

if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='Performs training for the WaifuNet.')
    parser.add_argument('-c', '--config', is_config_file=True, help='Configuration file path')

    waifunet.waifunet_parameters(parser)
    generator_network.generator_parameters(parser)
    discriminator_network.discriminator_parameters(parser)

    training_parameters(parser)

    args = parser.parse_args()

    do_training(args)
