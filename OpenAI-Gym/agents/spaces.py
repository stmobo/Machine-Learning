import gym
import numpy as np

def random_sample(distribution):
    return np.random.choice(distribution.shape[0], p=distribution)

# Converts from a single action output number to a number of discrete actions.
def discrete_group_index_to_action(index, groupings):
    t = index
    coords = []
    for sz in groupings:
        coords.append(int(int(t) % sz))
        t /= sz
    return coords

# Converts from discrete actions to a single action output number
def action_to_discrete_group_index(action, groupings):
    t = 0
    acc = 1
    for coord, sz in zip(action, groupings):
        t += (coord * acc)
        acc *= sz
    return t

# Converts a space to a set of "groupings" (discrete action spaces).
def get_discrete_groupings(space):
    groupings = []
    if isinstance(space, gym.spaces.discrete.Discrete):
        groupings.append(space.n)
    elif isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
        for lo, hi in zip(space.low, space.high):
            groupings.append(hi-lo+1)
    elif isinstance(space, gym.spaces.tuple_space.Tuple):
        groupings.extend(np.concatenate( [ get_discrete_groupings(s) for s in space.spaces ] ))
    return groupings

def get_continuous_output_size(space):
    size = 0
    if isinstance(space, gym.spaces.discrete.Discrete):
        size += 1
    elif isinstance(space, gym.spaces.box.Box):
        size += np.prod(space.shape)
    elif isinstance(space, gym.spaces.tuple_space.Tuple):
        size += np.sum( [ get_input_size(s) for s in space.spaces ] )
    return size

def output_to_continuous_action(space, action):
    if isinstance(space, gym.spaces.discrete.Discrete):
        return int(action)
    elif isinstance(space, gym.spaces.box.Box):
        return np.reshape(action, space.shape)
    elif isinstance(space, gym.spaces.tuple_space.Tuple):
        action_elems = []
        idx = 0
        for s in space.spaces:
            sz = get_continuous_output_size(s)
            action_elems.append( output_to_continuous_action(s, action[idx:idx+sz]) )
            idx += sz

        return tuple(action_elems)

def get_input_size(space):
    size = 0

    if isinstance(space, gym.spaces.discrete.Discrete):
        size += 1
    elif isinstance(space, gym.spaces.box.Box):
        size += np.prod(space.shape)
    elif isinstance(space, gym.spaces.tuple_space.Tuple):
        size += np.sum( [ get_input_size(s) for s in space.spaces ] )
    return size

def input_to_list(space, observation):
    if isinstance(space, gym.spaces.discrete.Discrete):
        return [ observation ]
    elif isinstance(space, gym.spaces.box.Box):
        return np.array(observation).flatten()
    elif isinstance(space, gym.spaces.tuple_space.Tuple):
        out = []
        for s, o in zip(space.spaces, observation):
            out.extend(input_to_list(s, o))
        return out
    raise TypeError("Input observation not of space type: " + str(type(observation)))
