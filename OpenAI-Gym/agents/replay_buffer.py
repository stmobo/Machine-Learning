from collections import deque
import numpy as np
import random

def future_rewards(r_list, gamma):
    future_r = []
    for i, reward in enumerate(r_list):
        # discounted sum of future rewards
        # maybe have the discount be gamma ** (i+t) instead of just t?
        future_r.append( np.sum(
            [ (gamma ** t) * f for t, f in enumerate(r_list[i:]) ]
        ) )
    return future_r
    
class ReplayBuffer:
    def __init__(self, buf_sz):
        self.buf_size = buf_sz
        self.buf_count = 0
        self.buf = deque()
    
    def append_experience(self, state, action, reward, terminal, next_state, dist):
        if self.buf_count < self.buf_size:
            self.buf_count += 1
        else:
            self.buf.popleft()
        self.buf.append( (state, action, reward, terminal, next_state, dist) )
    
    def append_rollout(self, rollout_data):
        if ("states" in rollout_data) and ("actions" in rollout_data) and ("rewards" in rollout_data) and ("terminals" in rollout_data) and ("next_states" in rollout_data) and ("dists" in rollout_data):
            for s, a, r, t, s2, ds in zip(rollout_data["states"], rollout_data["actions"], rollout_data["rewards"], rollout_data["terminals"], rollout_data["next_states"], rollout_data["dists"]):
                self.append_experience(s,a,r,t,s2, ds)
    
    def size(self):
        return self.buf_count
        
    def sample(self, n):
        sm = random.sample(self.buf, min(self.buf_count, n))
        
        return {
            "states": np.array( [exp[0] for exp in sm] ),
            "actions": np.array( [exp[1] for exp in sm] ),
            "rewards": np.array( [exp[2] for exp in sm] ),
            "terminals": np.array( [exp[3] for exp in sm] ),
            "next_states": np.array( [exp[4] for exp in sm] ),
            "dists": np.array( [exp[5] for exp in sm] ),
        }
        