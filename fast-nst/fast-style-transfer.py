import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import configargparse
import os
import sys

from fnst import training, image_transform_net, losses

# Main script code below:
parser = configargparse.ArgumentParser(description='Performs fast neural style transfer.')
parser.add_argument('-c', '--config', is_config_file=True, help='Configuration file path')

parser.add_argument('--checkpoint-dir', help='Directory where checkpoint files are stored.')

parser.add_argument('--distributed', action='store_true', help='Run in distributed mode')
parser.add_argument('--ps-hosts', action='append', help='List of parameter server hostname:port pairs')
parser.add_argument('--worker-hosts', action='append', help='List of worker node hostname:port pairs')
parser.add_argument('--job', help='Job name for this instance.')
parser.add_argument('--task-index', type=int, help='Task index for specified job.')

parser.add_argument('--vars-on-cpu', action='store_true', help='If passed, all variables will be pinned to main memory (instead of GPU memory).')

training.add_training_args(parser)
image_transform_net.add_network_hyperparameters(parser)
losses.add_loss_hyperparameters(parser)

args = parser.parse_args()

is_chief = True
if args.distributed:
    cluster = tf.train.ClusterSpec({'ps': args.ps_hosts, 'worker': args.worker_hosts})
    if args.job is None or args.task_index is None:
        hostname_parts = os.uname().nodename.split('-')
        job = hostname_parts[1]
        if job == 'chief':
            args.job = 'worker'
            args.task_index = 0
        elif job == 'worker' or job == 'ps':
            args.job = job
            args.task_index = int(hostname_parts[2])
        print("Autodetermined node role: Job \'{}\', index {:d}".format(args.job, args.task_index))
        sys.stdout.flush()

    server = tf.train.Server(cluster, job_name=args.job, task_index=args.task_index)
    is_chief = (args.job == 'worker' and args.task_index == 0)

    if args.job == 'ps':
        server.join()
        sys.exit(0)

if args.training:
    session_target = ''
    if args.distributed:
        session_target = server.target
        train_step, scaffold, hooks, chief_hooks, summary_writer, vgg_saver = training.setup_training(args, is_chief, server, cluster)
    else:
        var_device = ''
        if args.vars_on_cpu:
            var_device = '/cpu:0'

        with slim.arg_scope([slim.variable, slim.model_variable], device=var_device):
            train_step, scaffold, hooks, chief_hooks, summary_writer, vgg_saver = training.setup_training(args)

    with tf.train.MonitoredTrainingSession(
        master=session_target,
        is_chief=is_chief,
        hooks=hooks,
        chief_only_hooks=chief_hooks,
        scaffold=scaffold,
        save_summaries_steps=None,
        save_summaries_secs=None,
        save_checkpoint_secs=None,
    ) as mon_sess:
        training.init_vgg_model(args, mon_sess, vgg_saver)
        ckpt_path = tf.train.latest_checkpoint(args.checkpoint_dir)
        scaffold.saver.restore(mon_sess, ckpt_path)

        print('Beginning training...')
        sys.stdout.flush()

        local_run_steps = 0
        while not mon_sess.should_stop():
            if local_run_steps % args.session_trace_frequency == 0:
                run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_meta = tf.RunMetadata()

                retns = mon_sess.run([train_step, tf.contrib.framework.get_or_create_global_step()], options=run_opts, run_metadata=run_meta)

                summary_writer.add_run_metadata(run_meta, 'step-{}'.format(retns[1]))
                summary_writer.flush()
            else:
                retns = mon_sess.run([train_step, tf.contrib.framework.get_or_create_global_step()])

            if local_run_steps % args.console_output_frequency == 0:
                print("Finished step {}.".format(retns[1]))

            local_run_steps += 1
