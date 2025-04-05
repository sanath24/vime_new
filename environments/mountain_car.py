from environment import Environment
import gym

class MountainCarEnv(Environment):
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.state_dim = {
            'position': {
                'dtype': 'float32',
                'type': 'continuous'
            },
            'velocity': {
                'dtype': 'float32',
                'type': 'continuous'
            }
        }
        self.action_dim = {
            'direction': {
                'dtype': 'int',
                'type': 'discrete',
                'n_values': 3  # left, neutral, right
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
        return 2  # position and velocity
    
    def get_action_dim(self):
        return 3  # 3 discrete actions
    
    def get_policy_input_dim(self):
        return 2
    
    def get_policy_output_dim(self):
        return 3
    
    def get_model_input_dim(self):
        return 3  # state + action
    
    def get_model_output_dim(self):
        return 2 