from environment import Environment
import gym
import numpy as np
# from gym.wrappers import Monitor
# import os

class Walker2DEnv(Environment):
    def __init__(self, video_dir="video_output"):
        self.env = gym.make('Walker2d-v4')
        # os.makedirs(video_dir, exist_ok=True)  # Ensure the directory exists
        # self.env = Wrapper.Monitor(self.env, video_dir, force=True, video_callable=lambda episode_id: True)
        # self.env = Wrappers.Monitor(self.env, "video", force=True)
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
        # self.env.render()
        terminal = terminated or truncated
        return next_state, reward, terminal
    
    def get_state_dim(self):
        return 17  
    
    def get_action_dim(self):
        return 6  #
    
    def get_policy_input_dim(self):
        return 17
    
    def get_policy_output_dim(self):
        return 6
    
    def get_model_input_dim(self):
        return 18
    
    def get_model_output_dim(self):
        return 17
