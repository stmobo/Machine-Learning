#
# TRPO-DQN Agent: does TRPO updates with Q-values from DQN
#
import tensorflow as tf
import prettytensor as pt
import gym
import numpy as np
import csv
import os
import random

from collections import deque
from agents import dqn, trpo, spaces, replay_buffer

envName = 'CartPole-v0'
solvedThreshold = 195.0 # over 100 consecutive trials
env = gym.make(envName)

print("[DQN_TRPO-Agent-Tester] Observation space type: " + str(type(env.observation_space)))
print("[DQN_TRPO-Agent-Tester] Action space type: " + str(type(env.action_space)))

groupings = spaces.get_discrete_groupings(env.action_space)
inputShape = [ None, spaces.get_input_size(env.observation_space) ]
outputSize = int(np.prod(groupings))
outputShape = [None, outputSize]

print("[DQN_TRPO-Agent-Tester] Observation space shape: " + str(inputShape))
print("[DQN_TRPO-Agent-Tester] Output shape: " + str(outputShape))

render_frequency = 0 # set to 0 to disable rendering
batch_max_timesteps = 200
batch_n_episodes = 20
episode_max_time = 350
trial_max_episodes = 0

# General learning parameters:
discount_factor = 0.99

# DQN learning parameters:
learning_rate = 0.001
dqn_update_frequency = 4 # Number of timesteps to do in between parameter updates
gradient_clipping = (-1, 1)
dqn_batch_size = 64

# TRPO learning parameters:
divergence_constraint = 0.01
cg_damping = 0.1
trpo_update_frequency = 10
trpo_batch_size = 256

# Experience replay parameters:
replay_buffer_size = 1000000


# DQN Target network update parameters:
target_mix_factor = 0.001
target_update_frequency = 100 # number of policy updates to do before each target network update

# Epsilon-greedy parameters
epsilon_max = 1.0
epsilon_min = 0.1
epsilon_delta = 0.0001 # epsilon goes from 1.0 -> 0.0 over 1000 timesteps (~100 episodes)

# File / output parameters:
quiet_mode = False              # suppress output
no_summary = False              # suppress file output
flush_after_episode = True		# flush after every episode
output_dir = './data/dqn_trpo-' + envName
save_frequency = 100			# Save after N episodes
save_file_history = 15			# number of save files to keep

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def criticNetwork(inputTensor):
    return (pt.wrap(inputTensor)
        .fully_connected(300, activation_fn=tf.nn.relu)
        .fully_connected(300, activation_fn=tf.nn.relu)
        .fully_connected(300, activation_fn=tf.nn.relu)
        .fully_connected(outputSize, activation_fn=None))

def actorNetwork(inputTensor):
    net_out, loss = (pt.wrap(inputTensor)
        .fully_connected(300, activation_fn=tf.nn.relu)
        .fully_connected(300, activation_fn=tf.nn.relu)
        .fully_connected(300, activation_fn=tf.nn.relu)
        .softmax_classifier(outputSize))
    return net_out

def do_trial():
    graph = tf.get_default_graph()
    session = tf.Session()

    state_in = tf.placeholder(tf.float32, inputShape)

    with tf.variable_scope("critic"):
        critic = dqn.DeepQNetwork(graph, session, state_in, criticNetwork, \
            target_mix_factor=target_mix_factor, gradient_clipping=gradient_clipping,\
            discount_factor=discount_factor, learning_rate=learning_rate,\
            prefix="critic/")

    with tf.variable_scope("actor"):
        actor = trpo.TRPO(session, graph, state_in, actorNetwork, \
            prefix="actor/", divergence_constraint=divergence_constraint,\
            cg_damping = cg_damping)

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
        summary_file = tf.summary.FileWriter(summ_dirname, graph=graph, flush_secs=60)
        with graph.as_default():
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
    dqn_update_timer = dqn_update_frequency
    trpo_update_timer = trpo_update_frequency
    target_update_timer = target_update_frequency
    epsilon = epsilon_max
    experience_buffer = deque(maxlen=replay_buffer_size)

    while (trial_max_episodes == 0) or (episode < trial_max_episodes):
        timestep = 0
        observation = env.reset()
        doRender = False

        if render_frequency != 0 and episode % render_frequency == 0:
            doRender = True

        episode += 1
        episode_total_reward = 0

        critic_loss_values = []

        # Perform rollout:
        start_epsilon = epsilon
        episode_states = []

        while (episode_max_time == 0) or (timestep < episode_max_time):
            if doRender:
                env.render()

            net_in = np.expand_dims(spaces.input_to_list(env.observation_space, observation), axis=0)

            net_out = np.squeeze(actor.act(net_in))

            # Epsilon-greedy exploration:
            action = 0
            if (epsilon > 0) and (random.random() < epsilon):
                # Random action
                action = env.action_space.sample()
            else:
                # Pick action= argmax[a] from Q(s,a)
                action = np.squeeze(spaces.discrete_group_index_to_action(spaces.random_sample(net_out), groupings))

            next_observation, reward, done, info = env.step(action)

            experience_buffer.append( (observation, action, reward, done, next_observation, net_out) )
            episode_states.append(observation)

            # Do we need to do a DQN update?
            if (len(experience_buffer) > dqn_batch_size) and (dqn_update_timer <= 0):
                sm = random.sample(experience_buffer, dqn_batch_size)

                curInShape = inputShape[:]
                curInShape[0] = len(sm)

                states = np.reshape([ ts[0] for ts in sm ], curInShape)
                actions = np.reshape([ ts[1] for ts in sm ], [len(sm)])
                rewards = np.reshape([ ts[2] for ts in sm ], [len(sm)])
                term_state = np.reshape([ ts[3] for ts in sm ], [len(sm)])
                next_states = np.reshape([ ts[4] for ts in sm ], curInShape)

                # Do Deep Q-Learning first:
                critic_loss = critic.train(next_states, states, actions, term_state, rewards)

                # Reset parameter update timer
                dqn_update_timer = dqn_update_frequency

                # Do we need to update agent target networks?
                if target_update_timer <= 0:
                    target_update_timer = target_update_frequency
                    critic.target_copy()

                # Decrement target update timer:
                target_update_timer = max(0, target_update_timer-1)

                # Track Q-learning data
                critic_loss_values.append(critic_loss)

            observation = next_observation

            # Update timers, etc. for next TS
            timestep += 1
            episode_total_reward += reward
            epsilon = max(epsilon-epsilon_delta, epsilon_min)
            dqn_update_timer = max(0, dqn_update_timer-1)

            if done:
                break

        predicted_q = critic.predict_main(np.squeeze(episode_states))

        # Do a TRPO update:
        # Get critic network Q-values:
        if (len(experience_buffer) > trpo_batch_size) and (trpo_update_timer == 0):
            sm = random.sample(experience_buffer, trpo_batch_size)

            curInShape = inputShape[:]
            curOutShape = outputShape[:]
            curInShape[0] = curOutShape[0] = len(sm)

            states = np.reshape([ ts[0] for ts in sm ], curInShape)
            actions = np.reshape([ ts[1] for ts in sm ], [len(sm)])
            dists = np.reshape([ ts[5] for ts in sm ], curOutShape)

            critic_out = critic.predict_main(np.squeeze(states))

            q_values = []
            for q, a in zip(critic_out, actions):
                q_values.append(q[a])

            # standardize Q-values to have mean of 0 and stddev of 1:
            q_values -= np.mean(q_values)
            q_values /= np.std(q_values)

            # Do a TRPO update:
            actor.optimize_policy(states, actions, dists, np.squeeze(q_values))

            # Reset timer and container vars
            trpo_update_timer = trpo_update_frequency

        trpo_update_timer = max(0, trpo_update_timer-1)

        # Update episode data:
        episode_reward_set.append(episode_total_reward)
        if len(episode_reward_set) > 100:
            episode_reward_set = episode_reward_set[-100:]

        if not quiet_mode:
            print("Episode {}: T={}, R={} (mean R={}), E={}".format(episode, timestep, episode_total_reward, np.mean(episode_reward_set), start_epsilon))

        if not no_summary and (len(critic_loss_values) > 0):
            summary_file.add_summary(
            summOp.eval(
                {
                    'total_reward:0': episode_total_reward,
                    'avg_qmax:0': np.amax(predicted_q),
                    'avg_q:0': np.mean(predicted_q),
                    'avg_critic_loss:0': np.mean(critic_loss_values),
                    'cur_epsilon:0': start_epsilon
                },
                    session = session
                ),
                global_step=episode
            )

            if flush_after_episode:
                summary_file.flush()

            # Save on first episode and after all multiples of save_frequency:
            if (episode-1) % save_frequency == 0:
                saver.save(session, saveName, global_step=episode)

        if np.mean(episode_reward_set) >= solvedThreshold:
            print("=== ENVIRONMENT SOLVED ===")
            print("Required {} episodes.".format(episode))
            return

while True:
    do_trial()
