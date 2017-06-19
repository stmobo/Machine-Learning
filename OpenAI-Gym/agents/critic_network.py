import tensorflow as tf
import prettytensor as pt
import numpy as np
import gym
import math


tensorType = tf.float32

class CriticNetwork:
    def __init__(self, session, graph, sz_state, n_hidden, sz_hidden, act_func):
        self.session = session
        self.graph = graph

        with graph.as_default():
            self.in_shape = [None]
            self.input_state = tf.placeholder(tensorType, [None, sz_state],  name="critic_state")

            self.n_hidden_layers = n_hidden
            self.hidden_layer_size = sz_hidden
            self.activation_function = act_func

            self.discount_factor = 0.99
            self.target_mix_factor = 0.001

            self.main_net_ops()
            self.target_net_ops()
            self.critic_loss_ops()
            self.update_tgt_ops()

    def estimate_q(self, states, actions):
        return np.squeeze(self.session.run(
            self.mainNet,
            feed_dict={ self.input_state: states, self.input_action_main: actions }
        ))

    def update_target_parameters(self):
        return self.session.run( self.update_tgt )

    def estimate_target_q(self, next_states, target_actions):
        return self.session.run(
            self.tgtNet,
            feed_dict={ self.input_state: next_states, self.input_action_tgt: target_actions }
        )

    def train(self, states, td_targets, actions_main):
        return self.session.run(
            self.do_optimize,
            feed_dict={ self.input_state: states, self.target: np.squeeze(td_targets), self.input_action_main: actions_main }
        )

    ## functions to create TF ops

    def update_tgt_ops(self):
        tgt_params = tf.get_collection("critic_tgt_net")
        main_params = tf.get_collection("critic_main_net")

        self.update_tgt = [ t.assign(tf.mul(m, self.target_mix_factor) + tf.mul(t, 1-self.target_mix_factor)) for t, m in zip(tgt_params, main_params) ]

    def critic_loss_ops(self):
        self.target = tf.placeholder(tensorType, [None], name="critic_loss_target") #self.immediate_reward + (self.gamma*self.tgtNet)
        self.critic_loss = tf.reduce_mean( tf.square(self.target - self.mainNet) ) # mean squared error
        self.do_optimize = tf.train.AdamOptimizer().minimize(self.critic_loss, var_list = tf.get_collection("critic_main_net"))

    def main_net_ops(self):
        self.input_action_main = tf.placeholder(tensorType, self.in_shape, name="critic_action_main")
        with pt.defaults_scope(trainable_variables=True, variable_collections=[tf.GraphKeys.TRAINABLE_VARIABLES, "critic_main_net"]):
            self.mainNet = pt.wrap(tf.concat_v2( [self.input_state, tf.expand_dims(self.input_action_main, 1)], 1 ))
            for i in range(0, self.n_hidden_layers):
                self.mainNet = self.mainNet.fully_connected(self.hidden_layer_size, activation_fn=self.activation_function)
            self.mainNet = self.mainNet.fully_connected(1, activation_fn=None)

        main_params = tf.get_collection("critic_main_net")
        self.main_net_grad = tf.gradients(self.mainNet, main_params)
        self.action_grad = tf.gradients(self.mainNet, self.input_action_main)

    def target_net_ops(self):
        self.input_action_tgt = tf.placeholder(tensorType, self.in_shape, name="critic_action_tgt")
        with pt.defaults_scope(trainable_variables=True, variable_collections = [tf.GraphKeys.TRAINABLE_VARIABLES, "critic_tgt_net"]):
            self.tgtNet = pt.wrap(tf.concat_v2( [self.input_state, tf.expand_dims(self.input_action_tgt, 1)], 1 ))
            for i in range(0, self.n_hidden_layers):
                self.tgtNet = self.tgtNet.fully_connected(self.hidden_layer_size, activation_fn=self.activation_function)
            self.tgtNet = self.tgtNet.fully_connected(1, activation_fn=None)
