from policy import Policy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np

class PPOPolicy(Policy):
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, value_factor=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                 gae_lambda=0.95, epochs=4):
        super().__init__()
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_factor = value_factor
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr
        )
        
    def get_action(self, state):
        # Handle both single state and batched states
        is_single_state = len(state.shape) == 1
        if is_single_state:
            state = state.reshape(1, -1)
            
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs = self.policy_net(state_tensor)
        dist = distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if is_single_state:
            return action.item(), log_prob
        return action.numpy(), log_prob
    
    def evaluate_actions(self, states, actions) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = self.policy_net(states)
        dist = distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = torch.mean(dist.entropy())
        values = torch.squeeze(self.value_net(states))
        return log_probs, values, entropy
    
    def compute_gae(self, values, rewards, next_values, dones) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate Generalized Advantage Estimation
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values
        return advantages, returns
    
    def update(self, states, actions, rewards, next_states, log_probs, dones):        
        # states is a list of tensors, convert to single tensor
        
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        rewards = torch.cat(rewards)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        dones = torch.cat(dones)
        
        # convert to correct type
        rewards = rewards.float()
        dones = dones.float()
        states = states.float()
        next_states = next_states.float()
        actions = actions.float()
        log_probs = log_probs.float()
                
        # Compute values
        values = torch.squeeze(self.value_net(states))
        next_values = torch.squeeze(self.value_net(next_states))
        
        # Compute advantages using GAE
        advantages, returns = self.compute_gae(values, rewards, next_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimization loop
        for _ in range(self.epochs):
            # Get new evaluations
            new_log_probs, new_values, entropy = self.evaluate_actions(states, actions)
            
            # Compute ratio and clipped objective
            ratio = torch.exp(new_log_probs - log_probs.detach())
            surrogate1 = ratio * advantages.detach()
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
            
            # Compute losses
            policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))
            value_loss = nn.functional.mse_loss(new_values, returns.detach())
            entropy_loss = -entropy * self.entropy_coef
            
            # Total loss
            loss = policy_loss + self.value_factor * value_loss + entropy_loss
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                self.max_grad_norm
            )
            
            self.optimizer.step()