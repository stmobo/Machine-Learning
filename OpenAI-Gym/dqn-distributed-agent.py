import tensorflow as tf
import prettytensor as pt
import gym
import numpy as np
import csv
import os
import random

from collections import deque
from agents import dqn, spaces, replay_buffer

render_frequency = 0 # set to 0 to disable rendering
batch_max_timesteps = 200
batch_n_episodes = 20
episode_max_time = 5000
trial_max_episodes = 0

# General learning parameters:
learning_rate = 0.001
discount_factor = 0.99
parameter_update_frequency = 4 # Number of timesteps to do in between parameter updates
gradient_clipping = (-1, 1)

# Experience replay parameters:
replay_buffer_size = 1000000
replay_batch_size = 64

# Target network update parameters:
target_mix_factor = 0.001
target_update_frequency = 100 # number of policy updates to do before each target network update

# Misc. agent parameters
epsilon_max = 1.0
epsilon_min = 0.05
epsilon_delta = 0.0001 # epsilon goes from 1.0 -> 0.1 over 1000 timesteps (~100 episodes)
state_history_len = 4  # number of previous states to send to agent

# Environment parameters:
envName = 'CartPole-v0'
solvedThreshold = 195.0 # over 100 consecutive trials
inputSize = 2 * state_history_len
outputSize = 2

# File / output parameters:
quiet_mode = False              # suppress output
no_summary = False              # suppress file output
flush_after_episode = True		# flush after every episode
output_dir = './data/dqn-' + envName
save_frequency = 100			# Save after N episodes
save_file_history = 15			# number of save files to keep

# Networking:
cluster = tf.train.ClusterSpec({
    "ps": [],           # Stores / updates variables
    "worker": [],       # Does heavy computations
})
task_index = 0
# Note: the chief worker task also runs the environment and handles queues.

def createNetwork(inputTensor):
    return (pt.wrap(inputTensor)
        .fully_connected(300, activation_fn=tf.nn.relu)
        .fully_connected(300, activation_fn=tf.nn.relu)
        .fully_connected(300, activation_fn=tf.nn.relu)
        .fully_connected(outputSize, activation_fn=None))

# Creates ops to create the basic DQN network model and training ops.
# Executed by all workers.
#
# Optimize should be called as long as there are experiences in the queue (each gradient is averaged)
# mixnet is returned to allow for evaluation of actual agent env states (feed to mixnet.net_in)
# optimizer is returned for optimizer.get_chief_queue_runner() and maybe other stuff too
def modelOps(replayQueue):
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index), cluster=cluster):
        n_workers = len(cluster.as_dict()['worker'])
        experience = replayQueue.dequeue()

        state = experience[0]
        action = experience[1]
        reward = experience[2]
        term_state = experience[3]
        next_state = experience[4]

        mixnet = mixed_network.MixedNetwork(tf.get_default_graph(), states, \
            state, createNetwork, target_mix_factor=target_mix_factor,\
            target_net_in=next_state)

        # Compute TD-Targets Y[i]:
        # Y[i] = r[i] + (discount_factor * max(Q)) if s[i] is not terminal state
        # Y[i] = r[i] if s[i] is terminal state
        td_target = reward + (discount_factor * tf.reduce_max(mixnet.target_out) * term_state)

        # Compute loss:
        loss = tf.square(td_target - mixnet.main_out[action])

        optimizer = tf.train.SyncReplicasOptimizer(
            opt=tf.train.AdamOptimizer(learning_rate),          # Actual optimizer to use
            replicas_to_aggregate=n_workers*replay_batch_size,  # Total number of gradients to aggregate
            total_num_replicas=n_workers                        # Number of worker tasks (for barriers / synchronization)
        )

        optimize = None
        if isinstance(gradient_clipping, tuple):
            gradients = optimizer.compute_gradients(loss, mixnet.main_parameters)

            clipped_grads = [   \
            ( tf.clip_by_value(gv[0], gradient_clipping[0], gradient_clipping[1]), gv[1]) \
            for gv in gradients ]

            optimize = optimizer.apply_gradients(clipped_grads)
        else:
            optimize = optimizer.minimize(loss, var_list=mixnet.main_parameters)

        return optimize, mixnet, optimizer

def psServer(server):
