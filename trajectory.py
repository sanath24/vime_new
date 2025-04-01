import torch 
import numpy as np

class Trajectory:
    def __init__(self, start_state):
        self.states = [start_state]
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def add_step(self, action, next_state, reward, log_prob):
        self.states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def get_current_state(self):
        return self.states[-1]
    
    # TODO: Need a better name for this method
    def get_inputs_and_targets(self):
        states = torch.tensor(np.array(self.states[:-1]))
        actions = torch.tensor(np.array(self.actions))
        next_states = torch.tensor(np.array(self.states[1:]))
        rewards = torch.tensor(np.array(self.rewards))
        log_probs = torch.tensor(self.log_probs)
        
        return states, actions, next_states, rewards, log_probs