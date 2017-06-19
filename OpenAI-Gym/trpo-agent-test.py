import tensorflow as tf
import prettytensor as pt
import gym
import numpy as np
import csv
import os

from agents import trpo, spaces, replay_buffer

envName = 'CartPole-v0'
solvedThreshold = 195.0 # over 100 consecutive trials
env = gym.make(envName)

inputShape = [ None, spaces.get_input_size(env.observation_space) ]

print("[TRPO-Agent-Tester] Observation space type: " + str(type(env.observation_space)))
print("[TRPO-Agent-Tester] Observation space shape: " + str(inputShape))

print("[TRPO-Agent-Tester] Action space type: " + str(type(env.action_space)))

outputShape = spaces.get_discrete_groupings(env.action_space)
outputSize = np.prod(outputShape)
print("[TRPO-Agent-Tester] Output shape: " + str(outputShape))
print("[TRPO-Agent-Tester] Output size: " + str(outputSize))

net_in = tf.placeholder(trpo.tensorType, inputShape, name="net_in")
net_out, loss = (pt.wrap(net_in)
             .fully_connected(100, activation_fn=tf.tanh)
             .fully_connected(100, activation_fn=tf.tanh)
             .fully_connected(100, activation_fn=tf.tanh)
             .softmax_classifier(outputSize))

session = tf.Session()

agent = trpo.trpo_agent(session, tf.get_default_graph(), net_in, net_out, env, tf.trainable_variables())
agent.epsilon_delta = 0.001
agent.epsilon = 1.0
agent.epsilon_min = 0.1
agent.cg_damping = 0.1
agent.discount_factor = 0.95
agent.divergence_constraint = 0.01

render_frequency = 0 # set to 0 to disable rendering
batch_max_timesteps = 200
batch_n_episodes = 20
episode_max_time = 200

quiet_mode = False              # suppress output?
no_summary = False              # suppress file output
summarize_rendered = False      # Print summaries for rendered episodes?
batch_progress_report = False   # Print batch progress regularly (10%, 20%, ...)
per_episode_data = True         # Write data to file for each episode instead of per batch
per_episode_summary = False     # Print data to stdout for each episode instead of per batch

def future_rewards(r_list, gamma):
    future_r = []
    for i, reward in enumerate(r_list):
        # discounted sum of future rewards
        # maybe have the discount be gamma ** (i+t) instead of just t?
        future_r.append( np.sum(
            [ (gamma ** t) * f for t, f in enumerate(r_list[i:]) ]
        ) )
    return future_r

def run_trial():
    episode = 0
    batch = 0
    complete_reward_set = []
    session.run(tf.global_variables_initializer())


    while True:
        rollouts = []
        ep_lengths = []

        start_ep = episode
        fd_rewards = []

        current_batch_time = 0
        last_percent = 0.10

        while ((episode_max_time != 0) and ((episode - start_ep) < batch_n_episodes)) or ((episode_max_time == 0) and (current_batch_time < batch_max_timesteps)):
            doRender = False
            if (render_frequency != 0) and (episode % render_frequency == 0):
                doRender = True

            rollout_data = agent.rollout(episode_max_time, render=doRender)

            episode += 1
            rollouts.append(rollout_data)
            fd_rewards.append(future_rewards(rollout_data["rewards"], agent.discount_factor))

            ep_lengths.append(rollout_data["time"])
            current_batch_time += rollout_data["time"]

            if (episode_max_time == 0) and (round(current_batch_time / batch_max_timesteps, 1) > last_percent) and batch_progress_report:
                print("Batch processing {:%} complete...".format(round(current_batch_time / batch_max_timesteps, 1)))
                last_percent = round(current_batch_time / batch_max_timesteps, 1)

            if per_episode_summary or (doRender and summarize_rendered and not quiet_mode):
                print("Episode {}: Time: {}, Total R: {:f}, Mean R: {:f}".format(episode, rollout_data["time"], np.sum(rollout_data["rewards"]), np.mean(rollout_data["rewards"])))

            if not no_summary and per_episode_data:
                summary_out.writerow([episode, batch, rollout_data["time"], np.sum(rollout_data["rewards"]), np.mean(rollout_data["rewards"])])

        batch += 1

        summative_data = {
            # convert observations to actual inputs and then concatenate them all
            "states":  np.concatenate([ [ spaces.input_to_list(env.observation_space, s) for s in rollout["states"] ] for rollout in rollouts ]),
            "actions": np.concatenate([ rollout["actions"] for rollout in rollouts ]),
            "dists":   np.concatenate([ rollout["dists"] for rollout in rollouts ]),
            "info":   np.concatenate([ rollout["info"] for rollout in rollouts ]),
            "rewards": np.concatenate(fd_rewards)
        }

        # standardize rewards:
        summative_data["rewards"] -= summative_data["rewards"].mean()
        summative_data["rewards"] /= (summative_data["rewards"].std() + 1e-8) # 1e-8 == epsilon value to avoid divide-by-zero

        raw_rewards = np.concatenate([ rollout["rewards"] for rollout in rollouts ])
        total_rewards = [ np.sum(rollout["rewards"]) for rollout in rollouts ]

        complete_reward_set.extend(total_rewards)

        if len(complete_reward_set) > 100:
            complete_reward_set = complete_reward_set[-100:]

        if not quiet_mode:
            print("Episodes {} - {}: mean R: {:f} (std={:.3}), mean Q: {:.5} (std={:.5})".format(
                start_ep, episode, np.mean(total_rewards), np.std(total_rewards), summative_data["rewards"].mean(), summative_data["rewards"].std()))

        if not no_summary and not per_episode_data:
            summary_out.writerow([episode, total_rewards.mean(), total_rewards.std(), summative_data["rewards"].mean(), summative_data["rewards"].std()])

        if(np.mean(complete_reward_set) >= solvedThreshold):
            print("== ENVIRONMENT SOLVED ==")
            print("Required {} episodes.".format(episode))
            break

        agent.optimize_policy(summative_data)

while True:
    if not no_summary:
        trialn = 1
        while os.path.isfile("./data/trpo-"+envName+"-trial" + str(trialn) + ".csv"):
            trialn += 1

        filename = "./data/trpo-"+envName+"-trial" + str(trialn) + ".csv"

        print("Saving to: " + filename)
        with open(filename, "w", newline='') as summfile:
            summary_out = csv.writer(summfile)

            summary_out.writerow(["Epsilon-Max", "Epsilon-Min", "Epsilon-Delta", "KL Divergence Constraint", "Discount Factor"])
            summary_out.writerow([agent.epsilon, agent.epsilon_min, agent.epsilon_delta, agent.divergence_constraint, agent.discount_factor])
            if per_episode_data:
                summary_out.writerow(["Episode", "Batch", "Time", "Total Reward", "Mean Reward"])
            else:
                summary_out.writerow(["Batch End Episode", "Mean Total Reward", "Total Reward Std. Dev", "Mean Q-Value", "Q-Value Std. Dev"])

            run_trial()
    else:
        run_trial()
