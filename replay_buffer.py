import numpy as np
import torch

class ReplayBuffer:
    
    def __init__(self):
        self.buffer = []
    
    def add(self, state, action, next_state):
        self.buffer.append((state, action, next_state))
    
    def sample(self, n):
        # randomly sample n transitions from the buffer
        indices = np.random.choice(len(self.buffer), n, replace=True)
        
        states, actions, next_states = zip(*[self.buffer[i] for i in indices])
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        next_states = torch.tensor(next_states)
        return states, actions, next_states
    
    def clear(self):
        self.buffer = []
