from environment import Environment
from dm_control import suite
import numpy as np

class SwimmerEnv(Environment):
    def __init__(self):
        self.env = suite.load('swimmer', 'swimmer6')
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
        # print(type(time_step))
        # print(time_step)
        return np.concatenate([
            time_step.observation['joints'],
            time_step.observation['body_velocities'],
            time_step.observation['to_target']
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
        return 25  # 5 joints + 18 body velocities + 2 target position
    
    def get_action_dim(self):
        return 2  # 2D continuous action space (joint torques)
    
    def get_policy_input_dim(self):
        return 25  # Policy needs full state information (joints + velocities + target)
    
    def get_policy_output_dim(self):
        return 2  # Policy outputs 2 joint torques
    
    def get_model_input_dim(self):
        return 26  # state (25) + action (2)
    
    def get_model_output_dim(self):
        return 25  # Model predicts next state (same dimensions as state)