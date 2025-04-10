from policy import Policy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import os

class PPOPolicy(Policy):
    def __init__(self, state_dim, action_dim, device, action_space_type="discrete", 
                 hidden_dim=64, lr=3e-4, gamma=0.99, clip_epsilon=0.2, value_factor=0.5, 
                 entropy_coef=0.01, max_grad_norm=0.5, gae_lambda=0.95, epochs=4,
                 min_std=1e-6, init_std=1.0):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.action_space_type = action_space_type.lower()
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_factor = value_factor
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.min_std = min_std
        
        # Common feature extractor - Replaced BatchNorm with LayerNorm for better inference stability
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)
        
        # Policy network head depends on action space type
        if self.action_space_type == "discrete":
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            ).to(device)
        elif self.action_space_type == "continuous":
            # For continuous actions, we output means and log_stds
            self.mean_head = nn.Linear(hidden_dim, action_dim).to(device)
            self.log_std_head = nn.Parameter(torch.ones(action_dim) * np.log(init_std)).to(device)
        else:
            raise ValueError(f"Unsupported action space type: {self.action_space_type}. Use 'discrete' or 'continuous'.")
        
        # Value network - Replaced BatchNorm with LayerNorm
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Get all parameters for the optimizer
        if self.action_space_type == "discrete":
            policy_params = list(self.feature_extractor.parameters()) + list(self.policy_head.parameters())
        else:  # continuous
            policy_params = list(self.feature_extractor.parameters()) + list(self.mean_head.parameters()) + [self.log_std_head]
        
        # print total number of parameters
        total_params = sum(p.numel() for p in policy_params + list(self.value_net.parameters()))
        print(f"Total parameters in PPO policy: {total_params}")
        
        self.optimizer = optim.Adam(
            policy_params + list(self.value_net.parameters()), lr=lr
        )
        
    def get_action(self, state):
        # Handle both single state and batched states
        # Set all networks to eval mode
        self.feature_extractor.eval()
        self.value_net.eval()
        
        if self.action_space_type == "discrete":
            self.policy_head.eval()
        else:
            self.mean_head.eval()
        
        with torch.no_grad():  # No need to track gradients during inference
            is_single_state = len(state.shape) == 1
            if is_single_state:
                state = state.reshape(1, -1)
                
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            
            if self.action_space_type == "discrete":
                # Discrete action space
                features = self.feature_extractor(state_tensor)
                probs = self.policy_head(features)
                dist = distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                if is_single_state:
                    return action.item(), log_prob.item()
                return action.cpu().numpy(), log_prob.cpu()
            else:
                # Continuous action space
                features = self.feature_extractor(state_tensor)
                mean = self.mean_head(features)
                std = torch.exp(self.log_std_head.clamp(min=np.log(self.min_std)))
                dist = distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)  # Sum log probs across action dimensions
                
                if is_single_state:
                    return action.cpu().numpy()[0], log_prob.item()
                return action.cpu().numpy(), log_prob.cpu()
    
    def evaluate_actions(self, states, actions):
        features = self.feature_extractor(states)
        
        if self.action_space_type == "discrete":
            # Discrete action space
            probs = self.policy_head(features)
            dist = distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        else:
            # Continuous action space
            mean = self.mean_head(features)
            std = torch.exp(self.log_std_head.clamp(min=np.log(self.min_std)))
            dist = distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)  # Sum log probs across action dimensions
            entropy = dist.entropy().sum(dim=-1).mean()  # Mean of sum across action dimensions
        
        values = self.value_net(states).squeeze()
        return log_probs, values, entropy
    
    def compute_gae(self, values, rewards, next_values, dones):
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
        # Convert lists of tensors to single tensors
        states = torch.cat(states).to(self.device).float()
        next_states = torch.cat(next_states).to(self.device).float()
        rewards = torch.cat(rewards).to(self.device).float()
        log_probs = torch.cat(log_probs).to(self.device).float()
        dones = torch.cat(dones).to(self.device).float()
        
        # Handle actions differently based on action space type
        if self.action_space_type == "discrete":
            actions = torch.cat(actions).to(self.device).long()  # Discrete actions as long
        else:
            actions = torch.cat(actions).to(self.device).float()  # Continuous actions as float
        
        # Set networks to training mode
        self.feature_extractor.train()
        self.value_net.train()
        
        if self.action_space_type == "discrete":
            self.policy_head.train()
        else:
            self.mean_head.train()
                
        # Compute values for GAE
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            
            # Compute advantages using GAE
            advantages, returns = self.compute_gae(values, rewards, next_values, dones)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimization loop
        for _ in range(self.epochs):
            # Get new evaluations
            new_log_probs, new_values, entropy = self.evaluate_actions(states, actions)
            
            # Compute ratio and clipped objective
            ratio = torch.exp(new_log_probs - log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Compute losses
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = nn.functional.mse_loss(new_values, returns)
            entropy_loss = -entropy * self.entropy_coef
            
            # Total loss
            loss = policy_loss + self.value_factor * value_loss + entropy_loss
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            if self.action_space_type == "discrete":
                policy_params = list(self.feature_extractor.parameters()) + list(self.policy_head.parameters())
            else:
                policy_params = list(self.feature_extractor.parameters()) + list(self.mean_head.parameters()) + [self.log_std_head]
                
            nn.utils.clip_grad_norm_(
                policy_params + list(self.value_net.parameters()),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save model architecture and hyperparameters too
        save_dict = {
            'feature_extractor': self.feature_extractor.state_dict(),
            'value_net': self.value_net.state_dict(),
            'action_space_type': self.action_space_type,
            'hyperparams': {
                'gamma': self.gamma,
                'clip_epsilon': self.clip_epsilon,
                'gae_lambda': self.gae_lambda,
                'min_std': self.min_std
            }
        }
        
        if self.action_space_type == "discrete":
            save_dict['policy_head'] = self.policy_head.state_dict()
        else:
            save_dict['mean_head'] = self.mean_head.state_dict()
            save_dict['log_std_head'] = self.log_std_head
            
        torch.save(save_dict, os.path.join(path, "model.pth"))
    
    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, "model.pth"), map_location=self.device)
        
        # Check if action space type matches
        loaded_action_space = checkpoint['action_space_type']
        if loaded_action_space != self.action_space_type:
            raise ValueError(f"Loaded model has action space '{loaded_action_space}' but current model expects '{self.action_space_type}'")
            
        # Load network weights
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        
        # Load hyperparameters if available
        if 'hyperparams' in checkpoint:
            self.gamma = checkpoint['hyperparams'].get('gamma', self.gamma)
            self.clip_epsilon = checkpoint['hyperparams'].get('clip_epsilon', self.clip_epsilon)
            self.gae_lambda = checkpoint['hyperparams'].get('gae_lambda', self.gae_lambda)
            self.min_std = checkpoint['hyperparams'].get('min_std', self.min_std)
        
        if self.action_space_type == "discrete":
            self.policy_head.load_state_dict(checkpoint['policy_head'])
        else:
            self.mean_head.load_state_dict(checkpoint['mean_head'])
            self.log_std_head = checkpoint['log_std_head'].to(self.device)