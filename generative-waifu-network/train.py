import tensorflow as tf
import numpy as np
import waifunet
import configargparse

def training_parameters(parser):
    parser.add_argument('batch-size', type=int, default=16, help='Image batch size to use for training')
    parser.add_argument('input-queue-capacity', type=int, default=1000, help='Number of images to prefetch for the input queue')
    parser.add_argument('input-threads', type=int, default=4, help='Number of threads to use for input prefetching')
    parser.add_argument('dsc-steps', type=int, default=1, help='Number of discriminator / critic training steps to run for every generator training step')

    parser.add_argument('checkpoint-dir', help='Directory to write checkpoint files to')
    parser.add_argument('log-dir', help='Directory to write log files to')

def sample_pipeline(args):
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(args.content_image, name='input-filenames'), name='filename-producer')
    imreader = tf.TFRecordReader(name='image-reader')

    _, serialized_example = imreader.read(filename_queue)
    example = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.VarLenFeature(tf.string),
            'tags': tf.FixedLenFeature([args.label_size], tf.uint8)
        }
    )

    im_8u = tf.image.decode_image(example['image'], channels=3)
    im_f32 = tf.image.convert_image_dtype(im_8u, tf.float32)

    # Resize all images to match the generator's output dimensions
    img_resized = tf.image.resize_bicubic(img_f32, (args.output_height, args.output_width))
    img_resized = tf.reshape(imdata, [args.output_height, args.output_width, 3])

    # Scale all image values from range [0,1] to range [-1, 1] (same as TanH)
    img_out = (img_resized * 2.0) - 1.0

    # Convert tags to float tensors
    tags_f32 = tf.to_float32(example['tags'])

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

    if not args.wasserstein:
        shuffled_tags = tf.random_shuffle(sample_tags)
        dsc_mismatch_label = tf.random_uniform([], minval=0.0, maxval=0.3)
        mismatched_batch = tf.train.batch(
            [sample_images, shuffled_tags, dsc_mismatch_label]
            enqueue_many=True,
            batch_size=args.batch_size,
            capacity=args.input_queue_capacity,
            num_threads=args.input_threads,
            shared_name='mismatched-input-queue'
        )
    else:
        mismatched_batch = None

    sample_batch = (sample_images, sample_tags, sample_labels)

    # Noise values are sampled from a (truncated) normal distribution
    # and must fall within the range [0, 1]
    noise_batch = tf.truncated_normal([args.batch_size, args.z_size], mean=0.5, stddev=0.25)

    return sample_batch, noise_batch, mismatched_batch

def training_step(args, sess, summary_writer, wnet):
    for dsc_step in range(args.dsc_steps):
        _, dsc_summary = sess.run([wnet.dsc_train, wnet.self.dsc_summaries])
        summary_writer.add_summary(dsc_summary)

    _, gen_summary = sess.run([wnet.gen_train, wnet.gen_summaries])
    summary_writer.add_summary(gen_summary)

def do_training(args):
    sample_batch, noise_batch, mismatch_batch  = sample_pipeline(args)
    wnet = waifunet.waifunet(args, noise_batch, sample_batch[1], sample_batch, mismatch_batch)

    summ_writer = tf.summary.FileWriter(
        args.log_dir,
        graph=tf.get_default_graph()
    )

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=args.checkpoint_dir,
        save_summaries_steps=None,
    ) as mon_sess:
        while not mon_sess.should_stop():
            training_step(args, mon_sess, summ_writer, wnet)

if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='Performs training for the WaifuNet.')
    parser.add_argument('-c', '--config', is_config_file=True, help='Configuration file path')

    waifunet.waifunet_parameters(parser)
    add_training_args(parser)

    args = parser.parse_args()

    do_training(args)
