import tensorflow as tf
import prettytensor as pt
import gym
import numpy as np
import csv
import os
import random

from collections import deque
from agents import ddpg, spaces, replay_buffer

envName = 'Pendulum-v0'
solvedThreshold = -195.0 # over 100 consecutive trials
env = gym.make(envName)

print("[DDPG-Agent-Tester] Observation space type: " + str(type(env.observation_space)))
print("[DDPG-Agent-Tester] Action space type: " + str(type(env.action_space)))

inputShape = [ None, spaces.get_input_size(env.observation_space) ]
outputSize = spaces.get_continuous_output_size(env.action_space)
outputShape = [None, outputSize]

print("[DDPG-Agent-Tester] Observation space shape: " + str(inputShape))
print("[DDPG-Agent-Tester] Output shape: " + str(outputShape))

render_frequency = 0 # set to 0 to disable rendering
batch_max_timesteps = 200
batch_n_episodes = 20
episode_max_time = 1000
trial_max_episodes = 0

# General learning parameters:
actor_learning_rate = 0.001
critic_learning_rate = 0.01
discount_factor = 0.99
parameter_update_frequency = 0 # Number of timesteps to do in between parameter updates
actor_gradient_clipping = None
critic_gradient_clipping = (-1, 1)

# Experience replay parameters:
replay_buffer_size = 1000000
replay_batch_size = 64

# Target network update parameters:
actor_mix_factor = 0.001
critic_mix_factor = 0.001
target_update_frequency = 10000 # number of policy updates to do before each target network update

# Epsilon-greedy parameters
epsilon_max = 1.0
epsilon_min = 0.0
epsilon_delta = 0.000001 # epsilon goes from 1.0 -> 0.1 over 100,000 timesteps (100 episodes)

# File / output parameters:
quiet_mode = False              # suppress output
no_summary = False              # suppress file output
flush_after_episode = True		# flush after every episode
output_dir = './data/ddpg-' + envName
save_frequency = 100			# Save after N episodes
save_file_history = 15			# number of save files to keep

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

def actorNetwork(inputTensor):
	return (pt.wrap(inputTensor)
			  .fully_connected(300, activation_fn=tf.nn.relu)
			  .fully_connected(300, activation_fn=tf.nn.relu)
			  .fully_connected(300, activation_fn=tf.nn.relu)
			  .fully_connected(1, activation_fn=tf.tanh))

def criticNetwork(inputTensor):
	return (pt.wrap(inputTensor)
			  .fully_connected(300, activation_fn=tf.nn.relu)
			  .fully_connected(300, activation_fn=tf.nn.relu)
			  .fully_connected(300, activation_fn=tf.nn.relu)
			  .fully_connected(1, activation_fn=None))

def do_trial():
	agent = ddpg.DDPGAgent(actorNetwork, criticNetwork, inputShape, outputShape, \
				buf_sz=replay_buffer_size, batch_sz=replay_batch_size,
				critic_learning_rate=critic_learning_rate, \
				actor_learning_rate=actor_learning_rate,\
				actor_mix_factor=actor_mix_factor,\
				critic_mix_factor=critic_mix_factor,\
				discount_factor=discount_factor,\
				actor_gradient_clipping=actor_gradient_clipping,\
				critic_gradient_clipping=critic_gradient_clipping)

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
						]
					)

			# Create saver ops:
			saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=save_file_history)
			saveDir = os.path.join(summ_dirname, 'save')
			os.mkdir(saveDir)
			saveName = os.path.join(saveDir, 'model')

	if not quiet_mode:
		print("=== Beginning trial " + str(trial_n) + " ===")
		if not no_summary:
			print("Summary data saved to " + summ_dirname)

	# Episode tracking data (not used for running the agent/environment)
	episode = 0
	episode_reward_set = []

	# Values affecting the agent (learning, etc.)
	parameter_update_timer = parameter_update_frequency
	target_update_timer = target_update_frequency
	epsilon = epsilon_max

	while (trial_max_episodes == 0) or (episode < trial_max_episodes):
		timestep = 0
		observation = env.reset()
		doRender = False

		if render_frequency != 0 and episode % render_frequency == 0:
			doRender = True

		episode += 1
		episode_total_reward = 0

		qmax_values = []
		qavg_values = []
		critic_loss_values = []

		# Perform rollout:
		while (episode_max_time == 0) or (timestep < episode_max_time):
			if doRender:
				env.render()

			net_out = np.squeeze(agent.act(observation))

			# Epsilon-greedy exploration:
			action = 0
			if (epsilon > 0) and (random.random() < epsilon):
				action = env.action_space.sample()
			else:
				action = spaces.output_to_continuous_action(env.action_space, net_out*2) #+ (1.0 / (1.0 + episode + timestep))

			next_observation, reward, done, info = env.step(action) # note to self: remove the multiplier when using other envs

			agent.add_experience(observation, action, reward, done, next_observation)
			if (len(agent.replay_buf) > agent.batch_size) and (parameter_update_timer <= 0):
				predicted_q, critic_loss = agent.train()

				# Reset parameter update timer
				parameter_update_timer = parameter_update_frequency

				# Do we need to update agent target networks?
				if target_update_timer <= 0:
					target_update_timer = target_update_frequency
					agent.update_targets()

				# Decrement target update timer:
				target_update_timer = max(0, target_update_timer-1)

				# Track Q-learning data
				qmax_values.append(np.amax(predicted_q))
				qavg_values.append(np.mean(predicted_q))
				critic_loss_values.append(critic_loss)

			observation = next_observation

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
			print("Episode {}: T={}, R={}".format(episode, timestep, episode_total_reward))

		if not no_summary:
			summary_file.add_summary(
				summOp.eval(
					{
						'total_reward:0': episode_total_reward,
						'avg_qmax:0': np.mean(qmax_values),
						'avg_q:0': np.mean(qavg_values),
						'avg_critic_loss:0': np.mean(critic_loss_values)
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
	do_trial()
