import tensorflow as tf
import prettytensor as pt
import gym
import numpy as np
import csv
import os
import random

from collections import deque
from agents import dqn, spaces, replay_buffer


class Agent:
    def __init__(self, network)
