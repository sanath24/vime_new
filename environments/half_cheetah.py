from environment import Environment
import gym

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
        self.start_state = self.env.reset()
    
    def get_start_state(self):
        return self.start_state
    
    def reset(self):
        self.start_state, _ = self.env.reset()
    
    def step(self, action):
        next_state, reward, terminal, _, _ = self.env.step(action)
        return next_state, reward, terminal
    
    def get_state_dim(self):
        return 17  # 8 joint angles + 8 joint velocities + 1 position
    
    def get_action_dim(self):
        return 6  # 6D continuous action space
    
    def get_policy_input_dim(self):
        return 17
    
    def get_policy_output_dim(self):
        return 6
    
    def get_model_input_dim(self):
        return 23  # state + action
    
    def get_model_output_dim(self):
        return 17 