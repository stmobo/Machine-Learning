import numpy as np
import tensorflow as tf
import tflearn
import gym

from collections import deque
import configargparse
import random
import os

parser = configargparse.ArgumentParser(default_config_files=['dqnv2.conf'])

# Network input parameters
parser.add_argument('--state_history_len', env_var='STATE_HISTORY_LEN', type=int, default=4, help='Number of previous states to feed to the network')
parser.add_argument('--batch_size', env_var='BATCH_SIZE', type=int, default=32, help='Experience minibatch size used for training')
parser.add_argument('--buffer_size', env_var='BUFFER_SIZE', type=int, default=1000000, help='Experience buffer maximum size')

# Network training parameters
parser.add_argument('--learning_rate', env_var='LEARNING_RATE', type=float, default=0.001, help='Learning rate to use for training')
parser.add_argument('--regularizer_rate', env_var='REGULARIZER_RATE', type=float, default=0.01, help='Loss coefficient (beta) to use with L2 Regularization')
parser.add_argument('--primary_update_frequency', env_var='PRIMARY_UPDATE_FREQUENCY', type=int, default=4, help='How often, in timesteps, to update the main network parameters')
parser.add_argument('--target_update_frequency', env_var='TARGET_UPDATE_FREQUENCY', type=int, default=4000, help='How often, in timesteps, to update the target network parameters')
parser.add_argument('--min_gradient', env_var='MIN_GRADIENT', type=float, default=-1.0, help='Minimum allowed gradient during training (values below are clipped)')
parser.add_argument('--max_gradient', env_var='MAX_GRADIENT', type=float, default=1.0, help='Maximum allowed gradient during training (values below are clipped)')
parser.add_argument('--epsilon', env_var='EPSILON', type=float, default=0.05, help='Chance to perform a random action per timestep')

# General environment / runtime parameters
parser.add_argument('--solved_threshold', env_var='SOLVED_THRESHOLD', type=int, default=200, help='Average reward threshold to reach to solve the environment (over 100 trials)')
parser.add_argument('--log_basedir', env_var='LOG_BASEDIR', default='.\\data\\DQNv2-LunarLander-v2', help='Logging base directory; checkpoint and summary files will be placed in subdirectories here')
parser.add_argument('--checkpoint_period', env_var='CHECKPOINT_PERIOD', type=int, default=600, help='Time between checkpoint saves')
parser.add_argument('--render_frequency', env_var='RENDER_FREQUENCY', type=int, default=20, help='Render every N episodes')
parser.add_argument('--no_render', env_var='DISABLE_RENDER', action='store_true', help='Render every N episodes')

# Distributed mode parameters
parser.add_argument('--distributed', env_var='DISTRIBUTED_MODE', action='store_true', help='Runs the script in clustered mode')
parser.add_argument('--job', env_var='JOB_NAME', default='worker', help='Job type for this instance')
parser.add_argument('--task', env_var='TASK_INDEX', type=int, default=0, help='Task index for this instance')
parser.add_argument('--ps_hosts', env_var='PS_HOSTS', help='Parameter server host/port pairs, comma-separated')
parser.add_argument('--worker_hosts', env_var='WORKER_HOSTS', help='Worker host/port pairs, comma-separated')

args = parser.parse_args()

primary_network_prefix = 'main-net'
target_network_prefix = 'target-net'


state_size = 8  #
action_size = 4 #

# Holds network input Placeholders and output / training Tensors (when they're constructed)
network_inputs = {}
network_outputs = {}

# Q-Network Input State:
#  Should be a Tensor of shape [batch_len, timesteps, state_size*state_history_len]
# Returns a Tensor of shape [batch_len, action_space_len]
def q_network_model(input_state, scope_prefix='', reuse=False):
    with tf.variable_scope(scope_prefix, reuse=reuse):
        out = tflearn.layers.core.fully_connected(input_state, 4096, activation='relu', regularizer='L2', scope='fc1')
        out = tflearn.layers.core.fully_connected(out, 4096, activation='relu', regularizer='L2', scope='fc2')
        out = tflearn.layers.core.fully_connected(out, 4096, activation='relu', regularizer='L2', scope='fc3')
        out = tflearn.layers.core.fully_connected(out, 4096, activation='relu', regularizer='L2', scope='fc4')
        out = tflearn.layers.core.fully_connected(out, 4096, activation='relu', regularizer='L2', scope='fc5')
        out = tflearn.layers.core.fully_connected(out, 4096, activation='relu', regularizer='L2', scope='fc6')
        out = tflearn.layers.core.fully_connected(out, 4, activation='linear', regularizer='L2', scope='fc7')
        return out

def get_primary_network_parameters():
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=primary_network_prefix
    )

def get_target_network_parameters():
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=target_network_prefix
    )

# Creates ops to clone the primary network parameters to the target network
def clone_primary_to_target():
    primary_parameters = get_primary_network_parameters()
    target_parameters = get_target_network_parameters()

    clone_ops = [ target_parameters[i].assign(primary_parameters[i]) for i in range(len(target_parameters)) ]

    return tf.tuple(clone_ops)

# Builds the graph to perform Q-value prediction from one input state
def build_predict(input_state):
    q_values = q_network_model(input_state, scope_prefix=primary_network_prefix)
    tf.summary.scalar('Prediction Max Q', tf.reduce_max(q_values), collections=['predict_summary'])

    amax = tf.argmax(q_values, axis=1) # Reduce across action space Q values

    return amax, q_values

# Builds the graph to perform training for a batch of transitions
def build_train(states, actions, rewards, next_states, terminal):
    target_q_values = q_network_model(next_states, scope_prefix=target_network_prefix)
    predicted_q_values = q_network_model(states, scope_prefix=primary_network_prefix, reuse=True)

    targets = rewards
    max_qs = tf.reduce_max(target_q_values, axis=1)

    # Don't add target Q values to terminal states
    coeffs = tf.where(terminal, x=tf.ones_like(targets), y=tf.zeros_like(targets)) * args.learning_rate
    targets = targets + (coeffs * max_qs)

    # Get Q-values for each action in the batch:
    actions = tf.to_int32(actions)
    N = tf.range(0, tf.shape(actions)[0])

    indices = tf.stack([N, actions], axis=1) # Combine each action with its batch index no.
    action_q_values = tf.gather_nd(predicted_q_values, indices)

    # Optimize and regularize...
    model_loss =  tf.reduce_mean( tf.square( targets - action_q_values ) )
    reg_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = model_loss + (args.regularizer_rate * reg_loss)

    # Summarize losses
    tf.summary.scalar('Q-Value Loss', model_loss, collections=['train_summary'])
    tf.summary.scalar('Regularization Loss', reg_loss, collections=['train_summary'])
    tf.summary.scalar('Total Loss', loss, collections=['train_summary'])

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    gradients = optimizer.compute_gradients(loss, get_primary_network_parameters())

    # Summarize gradients
    mean_grads = []
    max_grads = []
    for g in gradients:
        if len(g) == 2:
            mean_grads.append(tf.reduce_mean(g[0]))
            max_grads.append(tf.reduce_max(g[0]))

    max_gradient = tf.reduce_max(max_grads)
    mean_gradient = tf.reduce_mean(mean_grads)
    tf.summary.scalar('Gradient Maximum', max_gradient, collections=['train_summary'])
    tf.summary.scalar('Gradient Mean', mean_gradient, collections=['train_summary'])

    # Clip and apply gradients
    clipped_gradients = [\
        (tf.clip_by_value(g[0], args.min_gradient, args.max_gradient), g[1]) \
        for g in gradients]

    train = optimizer.apply_gradients(clipped_gradients)

    return train, loss

# Creates everything in the graph.
def build_model():
    global network_inputs
    global network_outputs

    # We have two types of 'global step': one counting episodes, another counting raw timesteps
    episode_step = tf.Variable(0, name='global_episode_step')
    timestep = tf.Variable(0, name='global_step')

    network_inputs = {
        'states': tf.placeholder(tf.float32, shape=(None, state_size*args.state_history_len), name='input_states'),
        'actions': tf.placeholder(tf.int64, shape=(None), name='input_actions'),
        'rewards': tf.placeholder(tf.float32, shape=(None), name='input_rewards'),
        'next_states': tf.placeholder(tf.float32, shape=(None, state_size*args.state_history_len), name='input_next_states'),
        'terminal': tf.placeholder(tf.bool, shape=(None), name='input_terminal'),
        'episode_reward': tf.placeholder(tf.float32, name='input_episode_reward')
    }

    predict_out, predict_q_values = build_predict(network_inputs['states'])
    train, network_loss = build_train(
        network_inputs['states'],
        network_inputs['actions'],
        network_inputs['rewards'],
        network_inputs['next_states'],
        network_inputs['terminal'],
    )

    update_target = clone_primary_to_target()

    tf.summary.scalar('Episode Reward', network_inputs['episode_reward'], collections=['episode_summary'])

    network_outputs = {
        'predict_action': predict_out,
        'predict_q': predict_q_values,
        'train': train,
        'loss': network_loss,
        'update_target': update_target,
        'episode_summary': tf.summary.merge_all(key='episode_summary'),
        'predict_summary': tf.summary.merge_all(key='predict_summary'),
        'train_summary': tf.summary.merge_all(key='train_summary'),
        'get_episode_step': episode_step,
        'get_timestep': timestep,
        'inc_episode_step': tf.assign_add(episode_step, 1),
        'inc_timestep': tf.assign_add(timestep, 1)
    }

# Perform a single environment rollout
# sess is a session object
# env is a Gym environment
# exp_buffer is the network experience buffer
# sw is a summary writer, is_chief indicates whether we should write summaries
def rollout(sess, env, exp_buffer, sw, is_chief, should_render):
    global network_inputs
    global network_outputs

    current_state = env.reset()

    timestep = 0
    state_history = deque(
        [ np.zeros(state_size, np.float32) for i in range(args.state_history_len) ],
        maxlen=args.state_history_len
    )

    state_history.append(current_state)

    total_reward = 0

    while not sess.should_stop():
        if should_render:
            env.render()

        # Choose an action and execute it in the environment
        state_in = np.expand_dims(np.concatenate(list(state_history)), axis=0)
        predict_out = sess.run(
            [network_outputs['predict_action'], network_outputs['predict_summary'], network_outputs['inc_timestep']],
            feed_dict={network_inputs['states']: state_in}
        )

        # Add the prediction summary
        if is_chief:
            sw.add_summary(predict_out[1], global_step=predict_out[2])

        selected_action = predict_out[0]
        if random.random() < args.epsilon:
            action = env.action_space.sample()

        selected_action = np.squeeze(selected_action)

        next_state, reward, terminal_state, info = env.step(selected_action)

        # Get new state:
        state_history.append(next_state)
        next_state_in = np.expand_dims(np.concatenate(list(state_history)), axis=0)
        current_state = next_state

        # Store transition in experience buffer
        exp_buffer.append((state_in, selected_action, reward, terminal_state, next_state_in))

        if len(exp_buffer) > args.batch_size and predict_out[2] % args.primary_update_frequency == 0:
            # Sample from the experience buffer and train
            minibatch = random.sample(list(exp_buffer), min(len(exp_buffer), args.batch_size))

            states = np.reshape(
                [ ts[0] for ts in minibatch ],
                (len(minibatch), state_size * args.state_history_len)
            )

            next_states = np.reshape(
                [ ts[4] for ts in minibatch ],
                (len(minibatch), state_size * args.state_history_len)
            )

            actions = np.reshape([ ts[1] for ts in minibatch ], [len(minibatch)])
            rewards = np.reshape([ ts[2] for ts in minibatch ], [len(minibatch)])
            terminal = np.reshape([ ts[3] for ts in minibatch ], [len(minibatch)])

            retn = sess.run(
                [network_outputs['train'], network_outputs['loss'], network_outputs['train_summary']],
                feed_dict={
                    network_inputs['states']: states,
                    network_inputs['actions']: actions,
                    network_inputs['rewards']: rewards,
                    network_inputs['terminal']: terminal,
                    network_inputs['next_states']: next_states,
                }
            )

            if is_chief:
                # Write out the training summary
                sw.add_summary(retn[2], global_step=predict_out[2])

        # Update target network if necessary
        if predict_out[2] % args.target_update_frequency == 0:
            sess.run(network_outputs['update_target'])

        # Perform non-network bookkeeping:
        timestep += 1
        total_reward += reward

        if terminal_state:
            break
    return total_reward, timestep

# Finds an empty directory to put logging / checkpointing / summarization data under
def get_trial_number():
    trial_n = 1
    summ_dirname = os.path.join(args.log_basedir, 'trial-'+str(trial_n))
    while os.path.isdir(summ_dirname) or os.path.exists(summ_dirname):
        trial_n += 1
        summ_dirname = os.path.join(args.log_basedir, 'trial-'+str(trial_n))
    return trial_n, summ_dirname

# sess = some kind of ManagedSession object
# sw = summary writer
# is_chief = flag indicating whether we need to write summaries and stuff
def main_loop(sess, sw, is_chief):
    env = gym.make('LunarLander-v2')
    exp_buffer = deque(maxlen=args.buffer_size)

    episode_reward_history = deque(maxlen=100)
    while not sess.should_stop():
        episode_number = network_outputs['inc_episode_step'].eval(session=sess)
        should_render = True
        if args.render_frequency > 0:
            should_render = (episode_number % args.render_frequency == 0)
        if args.no_render:
            should_render = False
        episode_reward, episode_time = rollout(sess, env, exp_buffer, sw, is_chief, should_render)

        if is_chief:
            ep_summ = network_outputs['episode_summary'].eval(
                { network_inputs['episode_reward']: episode_reward },
                session=sess
            )

            sw.add_summary(ep_summ, global_step=episode_number)

            sw.flush()

        episode_reward_history.append(episode_reward)
        if np.mean(list(episode_reward_history)) > args.solved_threshold:
            sess.close()

        episode_number += 1

if not args.distributed:
    # Single computer mode
    build_model()

    trial_num, logdir = get_trial_number()

    scaffold = tf.train.Scaffold(saver=tf.train.Saver())
    hooks=[tf.train.CheckpointSaverHook(logdir, save_secs=args.checkpoint_period, scaffold=scaffold)]
    with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=logdir) as sess:
        summ_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
        main_loop(sess, summ_writer, True)
else:
    # Distributed mode
    worker_nodes = args.worker_hosts.split(',')
    ps_nodes = args.ps_hosts.split(',')

    cluster = tf.train.ClusterSpec({'ps': ps_nodes, 'worker': worker_nodes})
    server = tf.train.Server(cluster, job_name=args.job, task_index=args.task)
    is_chief = (args.job == 'worker' and args.task == 0)

    if args.job == 'ps':
        server.join()
    elif args.job == 'worker':
        with tf.device(tf.train.replica_device_setter(
            worker_device='/job:worker/task:%d' % args.task,
            cluster=cluster
        )):
            build_model()

        if is_chief:
            try:
                os.makedirs(args.log_basedir)
                print(os.path.exists(args.log_basedir))
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(path):
                    pass
                else:
                    raise

        with tf.train.MonitoredTrainingSession(
            master=server.target,
            is_chief=is_chief,
            checkpoint_dir=args.log_basedir,
            # Disable automatic summary saving (we handle that ourselves)
            save_summaries_steps=None,
            save_summaries_secs=None
        ) as mon_sess:
            summ_writer = tf.summary.FileWriter(args.log_basedir, graph=mon_sess.graph)
            main_loop(mon_sess, summ_writer, is_chief)
