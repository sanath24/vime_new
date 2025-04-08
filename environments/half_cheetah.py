from environment import Environment
import gym
import numpy as np

class HalfCheetahEnv(Environment):
    def __init__(self):
        self.env = gym.make('HalfCheetah-v4')
        self.state_dim = {
            'position': {
                'dtype': 'float32',
                'type': 'continuous'
            },
            'velocity': {
                'dtype': 'float32',
                'type': 'continuous'
            },
            'joint_angles': {
                'dtype': 'float32',
                'type': 'continuous'
            },
            'joint_velocities': {
                'dtype': 'float32',
                'type': 'continuous'
            }
        }
        self.action_dim = {
            'joint_torques': {
                'dtype': 'float32',
                'type': 'continuous',
                'shape': (6,)  # 6D continuous action space
            }
        }
        self.start_state = self.env.reset()[0]
    
    def get_start_state(self):
        return self.start_state
    
    def reset(self):
        self.start_state, _ = self.env.reset()
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.size == 1:
            action = np.full((6,), action.item(), dtype=np.float32)
        else:
            action = action.reshape(self.get_action_dim())
        next_state, reward, terminated, truncated, info = self.env.step(action)
        terminal = terminated or truncated
        return next_state, reward, terminal
    
    def get_state_dim(self):
        return 17  # Gym HalfCheetah-v4 observation space is of shape (17,)
    
    def get_action_dim(self):
        return 6  # 6D continuous action space
    
    def get_policy_input_dim(self):
        return 17
    
    def get_policy_output_dim(self):
        return 6
    
    def get_model_input_dim(self):
        return 18  # state (17) + action (6)
    
    def get_model_output_dim(self):
        return 17
