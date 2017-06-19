import tensorflow as tf
import numpy as np
import math

from agents import spaces

tensorType = tf.float32

#
# Utility class for converting from weight+bias matrices to/from flat vectors
#  of all the parameters.
#
class FlatConverter:
    def __init__(self, session, var_list):
        self.session = session                                  # So we can run ops
        self.var_list = var_list                                # references to the actual variables
        shapes = [v.get_shape().as_list() for v in var_list]           # Shapes for each variable
        sizes = [np.prod(shape) for shape in shapes]  # Total size in elements for each variable
        self.total_size = sum(sizes)

        self.flat = tf.placeholder(tensorType, [self.total_size])
        var_start = 0       # Current position in vector
        assignments = []    # list of assignment-group ops
        retrievals = []     # list of retrieval-group ops

        for (var, shape, size) in zip(var_list, shapes, sizes):
            # Define assignment operation (flat vector -> variables):
            assignments.append(tf.assign(
                var,
                tf.reshape(self.flat[var_start : var_start + size], shape)
            ))
            var_start += size

            # Define retrieval operation (variables -> flat vector):
            retrievals.append(tf.reshape(var, [-1]))

        self.set_op = tf.group(*assignments)
        self.get_op = tf.concat_v2(retrievals, 0)

    def set(self, flat):
        self.session.run(self.set_op, feed_dict={ self.flat: flat })

    def get(self):
        return self.session.run(self.get_op)

def tensor_list_flatten(in_list):
    return tf.concat_v2( [tf.reshape(elem, [-1]) for elem in in_list], 0 )

#
# Wraps arbitrary neural networks as TRPO agents.
#
class TRPO:
    def __init__(self, session, graph, net_in, netGen, \
                prefix="", divergence_constraint=0.01, cg_damping=0.1):
        with graph.as_default():
            self.session = session
            self.graph = graph
            self.input = net_in

            self.net = netGen(net_in)
            self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=prefix)

            self.flat_conv = FlatConverter(self.session, self.parameters)

            # various non-trainable and non-network parameters:
            self.divergence_constraint = divergence_constraint
            self.cg_damping = cg_damping

            self.fim_ops()
            self.loss_ops()

    def loss_ops(self):
        # Trajectory placeholders.
        # traj_state and traj_action form the usual (s, a) pairs.
        self.traj_action = tf.placeholder(tf.int32, [None], name="traj_action")             # traj. actions
        self.traj_dist = tf.placeholder(tensorType, self.net.get_shape().as_list(), name="traj_dist")    # traj. NNet outputs
        self.traj_adv = tf.placeholder(tensorType, [None], name="traj_adv")                 # traj. rewards / advantages
        self.traj_len = tf.shape(self.traj_action)[0]                                       # length of trajectory

        # Ops to calculate L(theta) (and its gradient)
        self.N = tf.range(0, self.traj_len)
        self.p_old = tf.gather_nd(self.traj_dist, tf.pack([self.N, self.traj_action], 1))
        self.p_new = tf.gather_nd(self.net, tf.pack([self.N, self.traj_action], 1))

        self.loss = -tf.reduce_sum( (self.p_new / self.p_old) * self.traj_adv )               # operation for loss function
        self.loss_gradient = tensor_list_flatten(tf.gradients(self.loss, self.parameters))    # gradient of loss function (stochastic policy gradient)

    # Calculate the FVP via reverse-mode AD of the KL-divergence
    def fim_ops(self):
        batch_sz_float = tf.cast(tf.shape(self.input)[0], tensorType)
        kl_divergence = tf.reduce_sum( tf.stop_gradient(self.net + 1e-8) * tf.log(tf.stop_gradient(self.net + 1e-8) / (self.net + 1e-8)) ) / batch_sz_float #tf.log(self.net)
        first_gradient = tf.gradients(kl_divergence, self.parameters)

        self.flat_tangents = tf.placeholder(tensorType, [None])
        tangents = []
        idx = 0
        for var in self.parameters:
            size = np.prod(var.get_shape().as_list())
            param = tf.reshape(self.flat_tangents[idx:idx+size], var.get_shape().as_list())
            tangents.append(param)
            idx += size

        self.fim = tensor_list_flatten(tf.gradients(
            [tf.reduce_sum(grad * tan) for (grad, tan) in zip(first_gradient, tangents)], self.parameters
        ))

    # Compute step direction using conjugate gradient method
    def step_dir(self, feed_val):
        g = -self.session.run(self.loss_gradient, feed_dict=feed_val)

        x = np.zeros_like( g )
        r = g.copy()
        p = g.copy()

        rsq_old = r.dot(r)
        for k in range(0,10):
            aP = self.fisher_vector_product(p, feed_val) + self.cg_damping * p
            pdot = p.dot(aP)

            if np.isnan(rsq_old):
                print("rsq_old is NaN?")
                break

            if np.isnan(pdot):
                print("pdot is NaN?")
                break

            alpha = rsq_old / (p.dot(aP) + 1e-8)

            x += (alpha * p)
            r -= (alpha * aP)

            rsq_new = r.dot(r)

            beta = (rsq_new / (rsq_old + 1e-8))
            p = r + (beta * p)

            rsq_old = rsq_new

            if rsq_new < 1e-8: # keep pdot from going to NaN
                break

            if np.isnan(rsq_new):
                print("rsq_new is NaN?")
                break

        return x

    # Optimize the current policy using line search.
    def optimize_policy(self, states, actions, dists, adv):
        feed = {
            self.input: states,
            self.traj_action: actions,
            self.traj_dist: dists,
            self.traj_adv: adv
        }

        s = self.step_dir(feed)
        step_As = s.dot(self.fisher_vector_product(s, feed))
        betasq = 2*self.divergence_constraint / (step_As + 1e-8)
        if betasq < 0:
            print("negative beta-squared ({})?".format(betasq))
            return
        beta = np.sqrt( betasq )

        current_theta = self.flat_conv.get()
        current_loss = self.session.run(self.loss, feed_dict=feed)

        best_theta = current_theta

        for k in range(0, 10):
            new_theta = current_theta + ((beta * (.5 ** k))*s)

            # compute new loss:
            self.flat_conv.set(new_theta)
            new_loss = self.session.run(self.loss, feed_dict=feed)

            if new_loss < current_loss:
                best_theta = new_theta
                current_loss = new_loss
                break

        self.flat_conv.set(best_theta)

    # Calculates a fisher-vector product.
    def fisher_vector_product(self, vec, feed_val):
        feed_val[self.flat_tangents] = vec
        return self.session.run(self.fim, feed_dict=feed_val)

    def act(self, state):
        return self.session.run(self.net, {self.input:state})
