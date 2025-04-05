from environment import Environment
from dm_control import suite
import numpy as np

class SwimmerGatherEnv(Environment):
    def __init__(self):
        self.env = suite.load('swimmer', 'gather')
        self.state_dim = {
            'joint_angles': {
                'dtype': 'float32',
                'type': 'continuous'
            },
            'joint_velocities': {
                'dtype': 'float32',
                'type': 'continuous'
            },
            'position': {
                'dtype': 'float32',
                'type': 'continuous'
            }
        }
        self.action_dim = {
            'joint_torques': {
                'dtype': 'float32',
                'type': 'continuous',
                'shape': (2,)  # 2D continuous action space
            }
        }
        self.start_state = self._get_state(self.env.reset())
    
    def _get_state(self, time_step):
        return np.concatenate([
            time_step.observation['joints'],
            time_step.observation['velocity'],
            time_step.observation['position']
        ])
    
    def get_start_state(self):
        return self.start_state
    
    def reset(self):
        time_step = self.env.reset()
        self.start_state = self._get_state(time_step)
    
    def step(self, action):
        time_step = self.env.step(action)
        next_state = self._get_state(time_step)
        reward = time_step.reward
        terminal = time_step.last()
        return next_state, reward, terminal
    
    def get_state_dim(self):
        return 8  # 2 joints + 2 velocities + 2 position + 2 orientation
    
    def get_action_dim(self):
        return 2  # 2D continuous action space
    
    def get_policy_input_dim(self):
        return 8
    
    def get_policy_output_dim(self):
        return 2
    
    def get_model_input_dim(self):
        return 10  # state + action
    
    def get_model_output_dim(self):
        return 8 