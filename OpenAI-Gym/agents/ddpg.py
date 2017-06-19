import tensorflow as tf
import prettytensor as pt
import numpy as np
import gym
import math
import random

from collections import deque
from agents import mixed_network, spaces, replay_buffer

tensorType = tf.float32

"""
Implements a Deep Deterministic Policy Gradient agent.

Adjustable parameters:
 - Actor / Critic learning rates
 - Temporal Difference discount factor
 - Experience Replay buffer / batch sizes
"""
class DDPGAgent:
    """
    Creates a new DDPG agent.

    Args:
    - actorGen and criticGen should be functions that create new
    neural networks with supplied Placeholder input Tensors.
    - state_shape will be the shape of the state input Placeholder.
    - action_shape should be the shape of the tensors output by the
    actor neural network.
    - buf_sz is the size of the agent's internal experience replay buffer.
    - batch_sz will be the size of each training batch (drawn from the replay buffer)
    """
    def __init__(self, actorGen, criticGen, state_shape, action_shape, buf_sz=100000,
                batch_sz=64, critic_learning_rate=0.001, actor_learning_rate=0.0001,
                discount_factor=0.99, actor_mix_factor=0.001,
                critic_mix_factor=0.001, actor_gradient_clipping=None, critic_gradient_clipping=None):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        self.discount_factor = discount_factor

        self.replay_buf = deque(maxlen=buf_sz)
        self.batch_size = batch_sz

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.__single_state_shape = self.state_shape[:]
        self.__single_state_shape[0] = 1

        with self.graph.as_default():
            self.state_in = tf.placeholder(tensorType, state_shape, name='state-in')
            self.action_in = tf.placeholder(tensorType, action_shape, name='action-in')

            with tf.variable_scope('critic'):
                self.critic = mixed_network.MixedNetwork(self.graph, self.session,
                                tf.concat_v2([self.state_in, self.action_in], axis=1),
                                criticGen, target_mix_factor=critic_mix_factor,
                                prefix='critic/')

                self.critic_prediction = tf.placeholder(tensorType, [None])
                self.critic_loss = tf.reduce_mean( tf.square( self.critic_prediction - tf.squeeze(self.critic.main_out) ) )
                critic_optimizer = tf.train.AdamOptimizer(critic_learning_rate)

                if isinstance(critic_gradient_clipping, tuple):
                    critic_gradients = critic_optimizer.compute_gradients(self.critic_loss, self.critic.main_parameters)

                    clipped_grads = [   \
                    ( tf.clip_by_value(gv[0], critic_gradient_clipping[0], critic_gradient_clipping[1]), gv[1]) \
                    for gv in critic_gradients ]

                    self.critic_optimize = critic_optimizer.apply_gradients(clipped_grads)
                else:
                    self.critic_optimize = critic_optimizer.minimize(self.critic_loss, var_list=self.critic.main_parameters)

                # gradient of the critic network w.r.t. the actions, averaged over all (s,a) pairs in batch
                self.action_gradient = tf.div(tf.gradients(self.critic.main_out, self.action_in), tf.constant(self.batch_size, tensorType))

            with tf.variable_scope('actor'):
                self.actor = mixed_network.MixedNetwork(self.graph,
                self.session, self.state_in, actorGen, prefix='actor/',
                target_mix_factor=actor_mix_factor)

                #self.aGrad_pl = tf.placeholder(tensorType, action_shape, name='action-gradient-placeholder')

                self.actor_gradients = tf.gradients(self.actor.main_out, self.actor.main_parameters, self.action_gradient)

                #self.actor_optimize = [p.assign(p + actor_learning_rate*g) \
                #for p, g in zip(self.actor.main_parameters, self.actor_gradients)]


                #self.actor_optimize = tf.train.GradientDescentOptimizer(actor_learning_rate).apply_gradients(
                #    zip(self.actor_gradients, self.actor.main_parameters)
                #)

                if isinstance(actor_gradient_clipping, tuple):
                    self.actor_gradients = [tf.clip_by_value(g, actor_gradient_clipping[0], actor_gradient_clipping[1]) for g in self.actor_gradients]

                self.actor_gradients = [tf.negative(g) for g in self.actor_gradients]

                self.actor_optimize = tf.train.AdamOptimizer(actor_learning_rate).apply_gradients(
                    zip(self.actor_gradients, self.actor.main_parameters)
                )

            self.session.run(tf.global_variables_initializer())

    def act(self, observation):
        return self.actor.get_main({ self.state_in: np.reshape(observation, self.__single_state_shape)})

    def add_experience(self, state, action, reward, done, next_state):
        self.replay_buf.append( (state, action, reward, done, next_state) )

    def train(self):
        sm = random.sample(self.replay_buf, min(len(self.replay_buf), self.batch_size))

        state_shape = self.state_shape[:]
        action_shape = self.action_shape[:]
        state_shape[0] = action_shape[0] = len(sm)

        states = np.reshape([ ts[0] for ts in sm ], state_shape)
        actions = np.reshape([ ts[1] for ts in sm ], action_shape)
        rewards = np.reshape([ ts[2] for ts in sm ], [len(sm)])
        term_state = np.reshape([ ts[3] for ts in sm ], [len(sm)])
        next_states = np.reshape([ ts[4] for ts in sm ], state_shape)

        # Use target actor and critic networks to estimate TD targets
        target_a =  np.reshape(self.actor.get_target({self.state_in:next_states}), action_shape)
        target_q =  np.reshape(self.critic.get_target({ self.state_in:next_states, self.action_in:target_a }), [len(sm)])

        td_targets = []
        for i, t in enumerate(target_q):
            if term_state[i]:
                td_targets.append(rewards[i])
            else:
                td_targets.append(rewards[i] + (self.discount_factor * t))

        _, crit_loss, predicted_q = self.session.run([self.critic_optimize, self.critic_loss, self.critic.main_out], {
            self.state_in: states,
            self.action_in: actions,
            self.critic_prediction: np.squeeze(td_targets)
        })

        net_actions = np.reshape(self.actor.get_main({self.state_in: states}), action_shape)
        self.session.run(self.actor_optimize, {self.state_in:states, self.action_in:net_actions})

        #self.session.run(self.actor_optimize, {self.state_in:states, self.action_in:actions})

        #actor_grad = self.session.run(self.actor_gradients, {self.state_in:states, self.action_in:net_actions})[0]
        #assert not np.isnan(np.sum(actor_grad))

        return np.squeeze(predicted_q), crit_loss

    def update_targets(self):
        self.actor.update_target()
        self.critic.update_target()
