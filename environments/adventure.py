from environment import Environment
import gym
import ale_py

class AdventureEnv(Environment):
     
    def __init__(self):
        self.env = gym.make('ALE/Adventure-v5', obs_type='ram')
        self.start_state = self.env.reset()
        
    def get_start_state(self):
        return self.start_state
    
    def reset(self):
        self.start_state, _ = self.env.reset()
        self.start_state = self.start_state.flatten()
    
    def step(self, action):
        next_state, reward, terminal, _, _ = self.env.step(action)
        
        # flatten the next_state
        next_state = next_state.flatten()
        
        return next_state, reward, terminal
    
    def get_state_dim(self):
        return 128
    
    def get_action_dim(self):
        return 18
    
    def get_policy_input_dim(self):
        return 128
    
    def get_policy_output_dim(self):
        return 18
    
    def get_model_input_dim(self):
        return 128 + 1
    
    def get_model_output_dim(self):
        return 128
    
    