class ReplayBuffer:
    
    def __init__(self):
        self.buffer = []
    
    def add(self, state, action, next_state):
        self.buffer.append((state, action, next_state))
    
    def sample(self, n):
        return self.buffer[:n]
