from policy import Policy

class DummyPolicy(Policy):
    
    def __init__(self):
        super().__init__()
        
    def get_action(self, state):
        return 0, 0
    
    def update(self, states, actions, rewards, next_states, log_probs):
        pass