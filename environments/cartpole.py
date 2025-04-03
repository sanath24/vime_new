from environment import Environment
import gym

class CartPoleEnv(Environment):
    
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.state_dim = {
            'cart_position': {
                'dtype': 'float32',
                'type': 'continuous'
            },
            'cart_velocity': {
                'dtype': 'float32',
                'type': 'continuous'
            },
            'pole_angle': {
                'dtype': 'float32',
                'type': 'continuous'
            },
            'pole_angular_velocity': {
                'dtype': 'float32',
                'type': 'continuous'
            }
        }
        self.action_dim = {
            'direction': {
                'dtype': 'int',
                'type': 'discrete',
                'n_values': 2
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
        return self.state_dim
    
    def get_action_dim(self):
        return self.action_dim
    
    def get_policy_input_dim(self):
        return 4
    
    def get_policy_output_dim(self):
        return 2
    
    def get_model_input_dim(self):
        return 5
    
    def get_model_output_dim(self):
        return 4