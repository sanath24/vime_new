import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os

class BNNLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, std_prior, device):
        super(BNNLayer, self).__init__()
        
        self.device = device
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.std_prior = std_prior
        self.rho_prior = self.std_to_log(self.std_prior)

        # Initialize directly on the target device
        self.W_mu = nn.Parameter(torch.randn(n_inputs, n_outputs, device=device))
        self.W_rho = nn.Parameter(torch.full((n_inputs, n_outputs), self.rho_prior, device=device))
        self.b_mu = nn.Parameter(torch.randn(n_outputs, device=device))
        self.b_rho = nn.Parameter(torch.full((n_outputs,), self.rho_prior, device=device))

        # Keep track of initial parameters (directly on device)
        self.W_mu_init = self.W_mu.clone().detach()
        self.W_rho_init = self.W_rho.clone().detach()
        self.b_mu_init = self.b_mu.clone().detach()
        self.b_rho_init = self.b_rho.clone().detach()

        # Old parameters for KL divergence (directly on device)
        self.W_mu_old = torch.zeros_like(self.W_mu, device=device)
        self.W_rho_old = torch.zeros_like(self.W_rho, device=device)
        self.b_mu_old = torch.zeros_like(self.b_mu, device=device)
        self.b_rho_old = torch.zeros_like(self.b_rho, device=device)

    def forward(self, X, infer=False):
        if infer:
            return torch.matmul(X, self.W_mu) + self.b_mu
        
        # Sample weights and biases during training
        W = self.get_W()
        b = self.get_b()
        
        return torch.matmul(X, W) + b

    def get_W(self):
        # Generate random noise directly on the device
        epsilon = torch.randn(self.n_inputs, self.n_outputs, device=self.device)
        return self.W_mu + self.log_to_std(self.W_rho) * epsilon

    def get_b(self):
        # Generate random noise directly on the device
        epsilon = torch.randn(self.n_outputs, device=self.device)
        return self.b_mu + self.log_to_std(self.b_rho) * epsilon

    def log_to_std(self, rho):
        return torch.log1p(torch.exp(rho))  # More numerically stable

    def std_to_log(self, std):
        return np.log(np.exp(std) - 1)

    def kl_div_new_prior(self):
        # Calculate KL divergence with respect to initial parameters
        kl_div = self.kl_div(
            self.W_mu, self.log_to_std(self.W_rho), 
            self.W_mu_init, self.log_to_std(self.W_rho_init))
        kl_div += self.kl_div(
            self.b_mu, self.log_to_std(self.b_rho), 
            self.b_mu_init, self.log_to_std(self.b_rho_init))
        return kl_div

    def kl_div_new_old(self):
        # Calculate KL divergence with respect to old parameters
        kl_div = self.kl_div(
            self.W_mu, self.log_to_std(self.W_rho), 
            self.W_mu_old, self.log_to_std(self.W_rho_old))
        kl_div += self.kl_div(
            self.b_mu, self.log_to_std(self.b_rho), 
            self.b_mu_old, self.log_to_std(self.b_rho_old))
        return kl_div

    def kl_div(self, p_mean, p_std, q_mean, q_std):
        # Vectorized KL divergence calculation
        numerator = (p_mean - q_mean).pow(2) + p_std.pow(2) - q_std.pow(2)
        denominator = 2 * q_std.pow(2) + 1e-8
        return (numerator / denominator + torch.log(q_std + 1e-8) - torch.log(p_std + 1e-8)).sum()

    def save_old_params(self):
        # Use in-place copy for efficiency
        self.W_mu_old.copy_(self.W_mu.data)
        self.W_rho_old.copy_(self.W_rho.data)
        self.b_mu_old.copy_(self.b_mu.data)
        self.b_rho_old.copy_(self.b_rho.data)

    def reset_to_old_params(self):
        # Use in-place copy for efficiency
        self.W_mu.data.copy_(self.W_mu_old)
        self.W_rho.data.copy_(self.W_rho_old)
        self.b_mu.data.copy_(self.b_mu_old)
        self.b_rho.data.copy_(self.b_rho_old)


class BNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, device, 
                 n_pred=5, num_replay_samples=100, epochs=10, 
                 kl_weight=0.01, lr=0.001):
        super(BNN, self).__init__()
        self.device = device
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_pred = n_pred
        self.num_replay_samples = num_replay_samples
        self.epochs = epochs
        self.kl_weight = kl_weight
        self.lr = lr
        
        print(f"KL weight: {self.kl_weight}")
        print(f"Learning rate: {self.lr}")
        
        # Build network directly on device
        self.layers = nn.ModuleList([
            BNNLayer(input_dim, hidden_dim, 0.1, device),
            nn.ReLU(),
            BNNLayer(hidden_dim, hidden_dim, 0.1, device),
            nn.ReLU(),
            BNNLayer(hidden_dim, output_dim, 0.1, device)
        ]).to(device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Loss function (pre-instantiate for efficiency)
        self.mse_loss = nn.MSELoss()

    def forward(self, x, infer=False):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        for layer in self.layers:
            if isinstance(layer, BNNLayer):
                x = layer(x, infer)
            else:
                x = layer(x)
        return x
    
    def infer(self, x):
        with torch.no_grad():  # No gradient tracking needed for inference
            return self.forward(x, infer=True)
    
    def kl_div_new_old(self):
        kl_div = 0
        for layer in self.layers:
            if isinstance(layer, BNNLayer):
                kl_div += layer.kl_div_new_old()
        return kl_div
    
    def kl_div_new_prior(self):
        kl_div = 0
        for layer in self.layers:
            if isinstance(layer, BNNLayer):
                kl_div += layer.kl_div_new_prior()
        return kl_div
    
    def loss(self, inputs, targets):
        # Move data to device if necessary
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Compute multiple forward passes for uncertainty estimation
        total_loss = 0
        for _ in range(self.n_pred):
            output = self.forward(inputs)
            total_loss += self.mse_loss(output, targets)
        
        # Average the prediction loss
        avg_pred_loss = total_loss / self.n_pred
        
        # Calculate KL divergence
        kl_div = self.kl_div_new_old()
        
        # Total loss with KL regularization
        total_loss = avg_pred_loss + self.kl_weight * kl_div
        
        return total_loss, avg_pred_loss, kl_div
    
    def loss_last_sample(self, input, target):
        # Ensure inputs have batch dimension
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
        
        # Ensure data is on device
        input = input.to(self.device)
        target = target.to(self.device)
        
        # Compute multiple forward passes
        total_loss = 0
        for _ in range(self.n_pred):
            output = self.forward(input)
            if len(output.shape) > 1 and output.shape[0] == 1 and len(target.shape) == 1:
                output = output.squeeze(0)
            total_loss += self.mse_loss(output, target)
        
        return total_loss / self.n_pred
    
    def eval_info_gain(self, inputs, targets):
        # Pre-allocate memory for results
        info_gain = torch.zeros(len(inputs), device=self.device)
        
        # Temporarily enable gradients even in eval mode
        with torch.enable_grad():
            for i in range(len(inputs)):
                # Save current parameters
                self.save_old_params()
                
                # Calculate loss for this sample
                loss = self.loss_last_sample(inputs[i], targets[i])
                
                # Perform gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Calculate KL divergence as information gain
                info_gain[i] = self.kl_div_new_old().detach()
                
                # Restore original parameters
                self.reset_to_old_params()
        
        return info_gain
    
    def reset_to_old_params(self):
        for layer in self.layers:
            if isinstance(layer, BNNLayer):
                layer.reset_to_old_params()
    
    def save_old_params(self):
        for layer in self.layers:
            if isinstance(layer, BNNLayer):
                layer.save_old_params()
    
    def update(self, replay_pool):
        # Save current parameters
        self.save_old_params()
        
        # Initialize tracking variables
        total_loss = 0
        total_sample_loss = 0
        total_divergence_loss = 0
        
        for _ in range(self.epochs):
            # Get batch from replay buffer
            states, actions, next_states = replay_pool.sample(self.num_replay_samples)
            
            # Move data to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            next_states = next_states.to(self.device)
            
            # Prepare inputs by concatenating states and actions
            inputs = torch.cat((states, actions.unsqueeze(1)), dim=1)
            
            # Standard precision training
            loss, sample_loss, divergence_loss = self.loss(inputs, next_states)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update totals
            total_loss += loss.item()
            total_sample_loss += sample_loss.item()
            total_divergence_loss += divergence_loss.item()
        
        # Save updated parameters
        self.save_old_params()
        
        return (total_loss / self.epochs, 
                total_sample_loss / self.epochs, 
                total_divergence_loss / self.epochs)

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), os.path.join(path, "bnn_model.pth"))
        
    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "bnn_model.pth"), 
                                        map_location=self.device))
        self.eval()