import tensorflow as tf
import numpy as np
from agents import mixed_network, spaces

tensorType = tf.float32

class DeepQNetwork:
    def __init__(self, graph, session, state_in, netGen, \
                target_mix_factor=0.001, gradient_clipping=None,\
                discount_factor=0.99, learning_rate=0.001, prefix=""):
        self.graph = graph
        self.session = session

        self.state_in = state_in
        self.discount_factor = discount_factor

        self.net = mixed_network.MixedNetwork(self.graph, self.session,\
                        self.state_in, netGen, target_mix_factor=target_mix_factor,\
                        prefix=prefix)

        self.targets = tf.placeholder(tensorType, [None])           # TD-targets Y[i]
        self.target_actions = tf.placeholder(tf.int32, [None])      # Actions to train on A[j]
        self.N = tf.range(0, tf.shape(self.target_actions)[0])

        self.net_q = tf.gather_nd(self.net.main_out, tf.pack([self.N, self.target_actions], 1)) # Q(s, A[j]) for all A[j] in minibatch
        self.loss = tf.reduce_mean( tf.square( self.targets - self.net_q ) )
        optimizer = tf.train.AdamOptimizer(learning_rate)

        if isinstance(gradient_clipping, tuple):
            gradients = optimizer.compute_gradients(self.loss, self.net.main_parameters)

            clipped_grads = [   \
            ( tf.clip_by_value(gv[0], gradient_clipping[0], gradient_clipping[1]), gv[1]) \
            for gv in gradients ]

            self.optimize = optimizer.apply_gradients(clipped_grads)
        else:
            self.optimize = optimizer.minimize(self.loss, var_list=self.net.main_parameters)

    def predict_main(self, state):
        return self.net.get_main({self.state_in:state})

    def predict_target(self, state):
        return self.net.get_target({self.state_in:state})

    def train(self, target_states, states, actions, term_state, rewards):
        target_q = self.net.get_target({ self.state_in:target_states })

        td_targets = []
        for i, t in enumerate(target_q):
            if term_state[i]:
                td_targets.append(rewards[i])
            else:
                td_targets.append(rewards[i] + (self.discount_factor * np.amax(t)))

        _, crit_loss = self.session.run([self.optimize, self.loss], {
            self.state_in: states,
            self.target_actions: np.squeeze(actions),
            self.targets: np.squeeze(td_targets)
        })

        return crit_loss

    def target_copy(self):
        return self.net.update_target()

    def target_mix(self):
        return self.net.mix_target()
