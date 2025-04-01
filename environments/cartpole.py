from environment import Environment
import gym

class CartPoleEnv(Environment):
    
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.state_dim = 4
        self.action_dim = 2
        self.start_state = self.env.reset()
    
    def get_start_state(self):
        return self.start_state
    
    def reset(self):
        self.start_state, _ = self.env.reset()
    
    def step(self, action):
        next_state, reward, terminal, _, _ = self.env.step(action)
        
        return next_state, reward, terminal
    
    def get_state_dim(self):
        return self.state_dim
    
    def get_action_dim(self):
        return self.action_dim