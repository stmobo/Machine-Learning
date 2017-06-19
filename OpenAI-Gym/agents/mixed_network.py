import tensorflow as tf

class MixedNetwork(object):
    """
    Implements a "mixed network" (main and target network pair).

    Args:

    - graph should be the Graph that both networks will be created in.

    - session should be a Session corresponding to graph.

    - net_in should be a Placeholder that will be used to feed input to
    both neural networks.

    - netGen should be a function that takes net_in, creates a neural network,
    and returns a Tensor net_out containing the network's output.

    """
    def __init__(self, graph, session, net_in, netGen, target_mix_factor=0.001, prefix='', target_net_in=None):
        self.session = session
        self.graph = graph

        with graph.as_default():
            self.net_in = net_in
            if target_net_in is None:
                self.target_net_in = self.net_in
            else:
                self.target_net_in = target_net_in

            # Make main net:
            with tf.variable_scope('main-net'):
                self.main_out = netGen(self.net_in)
                self.main_parameters = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=prefix+'main-net'
                )

            # Make target net:
            with tf.variable_scope('target-net'):
                self.target_out = netGen(self.target_net_in)
                self.target_parameters = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=prefix+'target-net'
                )

            # Mixer op:
            self.mix_tgt = [ self.target_parameters[i].assign( \
                tf.mul(self.main_parameters[i], target_mix_factor) + \
                tf.mul(self.target_parameters[i], 1.0-target_mix_factor)) \
                for i in range(len(self.target_parameters)) ]

            # Copy-update op:
            self.copy_tgt = [ self.target_parameters[i].assign( \
                              self.main_parameters[i]) \
                              for i in range(len(self.target_parameters)) ]

    def get_main(self, feed):
        return self.session.run(self.main_out, feed)

    def get_target(self, feed):
        return self.session.run(self.target_out, feed)

    def mix_target(self):
        self.session.run(self.mix_tgt)

    def update_target(self):
        self.session.run(self.copy_tgt)
