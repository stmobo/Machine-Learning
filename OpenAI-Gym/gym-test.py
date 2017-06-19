import gym
env = gym.make('CartPole-v0')

for ep_idx in range(1000):
    obs = env.reset()
    for ts in range(100):
        env.render()
        print(obs)
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        print(rew)
        if done:
            print("Episode finished after {} timesteps".format(ts+1))
            break
    