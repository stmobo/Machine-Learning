#
# Deep Q-Network agent
#

import tensorflow as tf
import prettytensor as pt
import gym
import numpy as np
import csv
import os
import random

from collections import deque
from agents import dqn, spaces, replay_buffer

envName = 'LunarLander-v2'
env = gym.make(envName)
solvedThreshold = 200.0 # over 100 consecutive trials

render_frequency = 50 # set to 0 to disable rendering
batch_max_timesteps = 200
batch_n_episodes = 20

episode_max_time = 0
timelimit_penalty = -500 # reward signal added if timelimit reached
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
state_history_len = 10  # number of previous states to send to agent

# File / output parameters:
quiet_mode = False              # suppress output
no_summary = False              # suppress file output
flush_after_episode = True		# flush after every episode
output_dir = './data/dqn-' + envName
save_frequency = 100			# Save after N episodes
save_file_history = 15			# number of save files to keep

# Remote computation:
remote_enabled = False
remote_server = "10.0.0.23:2222"    # host:port pair

cluster = None
server = None
session_target = ''
graph_target = ''
if remote_enabled:
    cluster = tf.train.ClusterSpec( {"remote": [remote_server], "local": ["localhost:2222"]} )
    server = tf.train.Server(cluster, job_name="local", task_index=0)
    session_target = "grpc://"+remote_server
    graph_target = '/job:remote/task:0'

print("[DQN-Agent-Tester] Observation space type: " + str(type(env.observation_space)))
print("[DQN-Agent-Tester] Action space type: " + str(type(env.action_space)))

groupings = spaces.get_discrete_groupings(env.action_space)
inputSize = spaces.get_input_size(env.observation_space)
inputShape = [ None, inputSize*state_history_len ]
outputSize = int(np.prod(groupings))
outputShape = [None, outputSize]

print("[DQN-Agent-Tester] Observation space shape: " + str(inputShape))
print("[DQN-Agent-Tester] Output shape: " + str(outputShape))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def createNetwork(inputTensor):
    net_out = (pt.wrap(inputTensor)
        .fully_connected(500, activation_fn=tf.nn.tanh)
        .fully_connected(500, activation_fn=tf.nn.tanh)
        .fully_connected(500, activation_fn=tf.nn.tanh)
        .fully_connected(500, activation_fn=tf.nn.tanh)
        .fully_connected(500, activation_fn=tf.nn.tanh)
        .fully_connected(outputSize, activation_fn=None))
    return tf.nn.softmax(net_out)

def do_trial(graph, session):
    state_in = tf.placeholder(dqn.tensorType, inputShape)

    agent = dqn.DeepQNetwork(graph, session, state_in, createNetwork, \
                target_mix_factor=target_mix_factor, gradient_clipping=gradient_clipping,\
                discount_factor=discount_factor, learning_rate=learning_rate)

    session.run(tf.global_variables_initializer())

    # Find a space to put summary data:
    summary_file = None
    summOp = None
    saver = None
    saveName = None
    if not no_summary:
        trial_n = 1
        summ_dirname = os.path.join(output_dir, 'trial-'+str(trial_n))
        while os.path.isdir(summ_dirname) or os.path.exists(summ_dirname):
            trial_n += 1
            summ_dirname = os.path.join(output_dir, 'trial-'+str(trial_n))
        os.mkdir(summ_dirname)

        # Create summary ops:
        summary_file = tf.summary.FileWriter(summ_dirname, graph=agent.graph, flush_secs=60)
        with agent.graph.as_default():
            # This op is called further below (after the episode is complete)
            summOp = tf.summary.merge(
                [
                    tf.summary.scalar('Total Reward', tf.placeholder(tf.float32, name='total_reward')),
                    tf.summary.scalar('Predicted Qmax', tf.placeholder(tf.float32, name='avg_qmax')),
                    tf.summary.scalar('Predicted Q', tf.placeholder(tf.float32, name='avg_q')),
                    tf.summary.scalar('Critic Loss', tf.placeholder(tf.float32, name='avg_critic_loss')),
                    tf.summary.scalar('Current Epsilon', tf.placeholder(tf.float32, name='cur_epsilon')),
                ]
            )

            # Create saver ops:
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=save_file_history)
            saveDir = os.path.join(summ_dirname, 'save')
            os.mkdir(saveDir)
            saveName = os.path.join(saveDir, 'model')
            print("=== Beginning trial " + str(trial_n) + " ===")
            print("Summary data saved to " + summ_dirname)

    # Episode tracking data (not used for running the agent/environment)
    episode = 0
    episode_reward_set = []

    # Values affecting the agent (learning, etc.)
    parameter_update_timer = parameter_update_frequency
    target_update_timer = target_update_frequency
    epsilon = epsilon_max
    experience_buffer = deque(maxlen=replay_buffer_size)

    while (trial_max_episodes == 0) or (episode < trial_max_episodes):
        timestep = 0
        observation = env.reset()
        doRender = False

        if (render_frequency != 0 and episode % render_frequency == 0):
            doRender = True

        episode += 1
        episode_total_reward = 0

        qmax_values = []
        qavg_values = []
        critic_loss_values = []

        # init agent with empty (zero) states and initial state
        state_history = deque([ np.zeros(inputSize, np.float32) for i in range(state_history_len) ], maxlen=state_history_len)
        state_history.append(spaces.input_to_list(env.observation_space, observation))

        # Perform rollout:
        start_epsilon = epsilon
        while (episode_max_time == 0) or (timestep < episode_max_time):
            if (timestep > 250) or doRender:
                env.render()

            # Feed current buffer to the agent
            state_in = np.expand_dims(np.concatenate(list(state_history)), axis=0)
            net_out = np.squeeze(agent.predict_main(state_in))

            # Epsilon-greedy exploration:
            action = 0
            if (epsilon > 0) and (random.random() < epsilon):
                # Random action
                action = env.action_space.sample()
            else:
                # Pick action= argmax[a] from Q(s,a)
                action = np.squeeze(spaces.discrete_group_index_to_action(np.argmax(net_out), groupings))

            next_observation, reward, done, info = env.step(action) # note to self: remove the multiplier when using other envs

            # modify last TS reward if necessary:
            if (not (episode_max_time == 0)) and (timestep == (episode_max_time - 1)):
                reward += timelimit_penalty

            # Copy in new state:
            observation = next_observation
            state_history.append(spaces.input_to_list(env.observation_space, observation))
            next_history = np.expand_dims(np.concatenate(list(state_history)), axis=0)

            experience_buffer.append( (state_in, action, reward, done, next_history) )

            if (len(experience_buffer) > replay_batch_size) and (parameter_update_timer <= 0):
                sm = random.sample(experience_buffer, min(len(experience_buffer), replay_batch_size))

                curInShape = inputShape[:]

                curInShape[0] = len(sm)

                states = np.reshape([ ts[0] for ts in sm ], curInShape)
                actions = np.reshape([ ts[1] for ts in sm ], [len(sm)])
                rewards = np.reshape([ ts[2] for ts in sm ], [len(sm)])
                term_state = np.reshape([ ts[3] for ts in sm ], [len(sm)])
                next_states = np.reshape([ ts[4] for ts in sm ], curInShape)

                critic_loss = agent.train(next_states, states, actions, term_state, rewards)

                # Reset parameter update timer
                parameter_update_timer = parameter_update_frequency

                # Do we need to update agent target networks?
                if target_update_timer <= 0:
                    target_update_timer = target_update_frequency
                    agent.target_copy()

                # Decrement target update timer:
                target_update_timer = max(0, target_update_timer-1)

                # Track Q-learning data
                qmax_values.append(np.amax(net_out))
                qavg_values.append(np.mean(net_out))
                critic_loss_values.append(critic_loss)

            # Update timers, etc. for next TS
            timestep += 1
            episode_total_reward += reward
            epsilon = max(epsilon-epsilon_delta, epsilon_min)
            parameter_update_timer = max(0, parameter_update_timer-1)

            if done:
                break

        # Update episode data:
        episode_reward_set.append(episode_total_reward)
        if len(episode_reward_set) > 100:
            episode_reward_set = episode_reward_set[-100:]

        if not quiet_mode:
            print("Episode {}: T={}, R={} (mean R={}), E={}".format(episode, timestep, episode_total_reward, np.mean(episode_reward_set), start_epsilon))

        if not no_summary and (len(qmax_values) > 0 and len(qavg_values) > 0 and len(critic_loss_values) > 0):
            summary_file.add_summary(
            summOp.eval(
                {
                    'total_reward:0': episode_total_reward,
                    'avg_qmax:0': np.mean(qmax_values),
                    'avg_q:0': np.mean(qavg_values),
                    'avg_critic_loss:0': np.mean(critic_loss_values),
                    'cur_epsilon:0': start_epsilon
                },
                    session = agent.session
                ),
                global_step=episode
            )

            if flush_after_episode:
                summary_file.flush()

            # Save on first episode and after all multiples of save_frequency:
            if (episode-1) % save_frequency == 0:
                saver.save(agent.session, saveName, global_step=episode)

        if np.mean(episode_reward_set) >= solvedThreshold:
            print("=== ENVIRONMENT SOLVED ===")
            print("Required {} episodes.".format(episode))
            return

while True:
    g = tf.Graph()
    with g.as_default():
        with tf.device(graph_target), tf.Session(session_target) as session:
            do_trial(g, session)
