import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os
import torch.nn.functional as F

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

        # Keep track of prior parameters (directly on device)
        self.W_mu_prior = self.W_mu.clone().detach()
        self.W_rho_prior = self.W_rho.clone().detach()
        self.b_mu_prior = self.b_mu.clone().detach()
        self.b_rho_prior = self.b_rho.clone().detach()

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
        # convert X to float tensor if it's not already
        if not isinstance(X, torch.FloatTensor):
            X = X.float()
        
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
            self.W_mu_prior, self.log_to_std(self.W_rho_prior))
        kl_div += self.kl_div(
            self.b_mu, self.log_to_std(self.b_rho), 
            self.b_mu_prior, self.log_to_std(self.b_rho_prior))
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
        
    def save_prior_params(self):
        # Save prior parameters for KL divergence calculation
        self.W_mu_prior = self.W_mu.clone().detach()
        self.W_rho_prior = self.W_rho.clone().detach()
        self.b_mu_prior = self.b_mu.clone().detach()
        self.b_rho_prior = self.b_rho.clone().detach()

    def reset_to_old_params(self):
        # Use in-place copy for efficiency
        self.W_mu.data.copy_(self.W_mu_old)
        self.W_rho.data.copy_(self.W_rho_old)
        self.b_mu.data.copy_(self.b_mu_old)
        self.b_rho.data.copy_(self.b_rho_old)


class BNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, device, 
                 n_pred=5, batch_size=100, epochs=10, 
                 kl_weight=0.01, lr=0.001, min_std=0.01):
        super(BNN, self).__init__()
        self.device = device
        
        self.kl_div_hist = []
        self.old_kl_div_hist = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_pred = n_pred
        self.batch_size = batch_size
        self.epochs = epochs
        self.kl_weight = kl_weight
        self.lr = lr
        self.min_std = min_std  # Minimum standard deviation for numerical stability
        
        print(f"KL weight: {self.kl_weight}")
        print(f"Learning rate: {self.lr}")
        
        # Build network directly on device
        # Feature extractor shared by mean and std networks
        self.feature_extractor = nn.Sequential(
            BNNLayer(input_dim, hidden_dim, 1, device),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            BNNLayer(hidden_dim, hidden_dim, 1, device),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ).to(device)
        
        # Mean prediction head
        self.mean_head = BNNLayer(hidden_dim, output_dim, 0.1, device).to(device)
        
        # Log standard deviation prediction head
        self.log_std_head = BNNLayer(hidden_dim, output_dim, 1, device).to(device)
        
        # Add all modules to a list for convenience
        self.all_layers = [
            self.feature_extractor[0],
            self.feature_extractor[3],
            self.mean_head,
            self.log_std_head
        ]
        # Print total number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params}")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, infer=False):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Extract features
        features = x
        for layer in self.feature_extractor:
            if isinstance(layer, BNNLayer):
                features = layer(features, infer)
            else:
                features = layer(features)
        
        # Predict mean
        mean = self.mean_head(features, infer)
        
        # Predict log standard deviation
        log_std = self.log_std_head(features, infer)
        
        # Apply soft plus and add minimum std for numerical stability
        std = F.softplus(log_std) + self.min_std
        
        return mean, std
    
    def infer(self, x):
        with torch.no_grad():  # No gradient tracking needed for inference
            mean, std = self.forward(x, infer=True)
            return mean, std
    
    def gaussian_nll_loss(self, mean, std, target):
        """
        Gaussian Negative Log-Likelihood loss with clamping on variance
        """
        # Set a minimum value for standard deviation to prevent numerical instability
        min_std = 1e-6  # You can adjust this value based on your needs
        
        # Clamp the standard deviation to be at least `min_std`
        std = torch.clamp(std, min=min_std)

        # Calculate negative log likelihood
        # NLL = 0.5 * log(2π) + log(σ) + 0.5 * ((y - μ) / σ)²
        nll = (torch.log(std) + 0.5 * torch.log(2 * torch.tensor(np.pi, device=self.device)) + 
            0.5 * ((target - mean) / std).pow(2)).mean()

        return nll

    
    def kl_div_new_old(self):
        kl_div = 0
        for layer in self.all_layers:
            kl_div += layer.kl_div_new_old()
        return kl_div
    
    def kl_div_new_prior(self):
        kl_div = 0
        for layer in self.all_layers:
            kl_div += layer.kl_div_new_prior()
        return kl_div
    
    def loss(self, inputs, targets, kl_div_prior=False):
        # Move data to device if necessary
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Compute multiple forward passes for uncertainty estimation
        total_loss = 0
        for _ in range(self.n_pred):
            mean, std = self.forward(inputs)
            total_loss += self.gaussian_nll_loss(mean, std, targets)
        
        # Average the prediction loss
        avg_pred_loss = total_loss / self.n_pred
        
        # Calculate KL divergence
        if kl_div_prior:
            kl_div = self.kl_div_new_prior()
        else:
            kl_div = self.kl_div_new_old()
            
        
        # Total loss with KL regularization
        total_loss = avg_pred_loss + kl_div
        
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
            mean, std = self.forward(input)
            if len(mean.shape) > 1 and mean.shape[0] == 1 and len(target.shape) == 1:
                mean = mean.squeeze(0)
                std = std.squeeze(0)
            total_loss += self.gaussian_nll_loss(mean, std, target)
        
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
    
    def save_prior_params(self):
        for layer in self.all_layers:
            if isinstance(layer, BNNLayer):
                layer.save_prior_params()
        
    def eval_info_gain_and_update(self, inputs, targets):
        # divide inputs and targets into batches
        num_batches = len(inputs) // self.batch_size
        info_gain = torch.zeros(len(inputs), device=self.device)
        total_loss = 0
        total_sample_loss = 0
        total_divergence_loss = 0
        avg_kl_div = 0
        if len(self.old_kl_div_hist) > 0:
            avg_kl_div = np.mean(self.old_kl_div_hist)
            
        for i in range(num_batches):
            self.save_old_params()
            batch_inputs = inputs[i * self.batch_size: min((i + 1) * self.batch_size, len(inputs))]
            batch_targets = targets[i * self.batch_size: min((i + 1) * self.batch_size, len(inputs))]
            loss, sample_loss, divergence_loss = self.loss(batch_inputs, batch_targets, True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_sample_loss += sample_loss.item()
            total_divergence_loss += divergence_loss.item()
            kl_div = self.kl_div_new_old().item()
            self.kl_div_hist.append(kl_div)
            info_gain[min((i + 1) * self.batch_size - 1, len(inputs) - 1)] = kl_div - avg_kl_div
        
        return info_gain, (total_loss / num_batches), (total_sample_loss / num_batches), (total_divergence_loss / num_batches)
            
    
    def reset_to_old_params(self):
        for layer in self.all_layers:
            if isinstance(layer, BNNLayer):
                layer.reset_to_old_params()
    
    def save_old_params(self):
        for layer in self.all_layers:
            if isinstance(layer, BNNLayer):
                layer.save_old_params()
    
    def reset_kl_div_hist(self):
        self.old_kl_div_hist = self.kl_div_hist.copy()
        self.kl_div_hist = []
    
    def update(self, replay_pool):
        # Save current parameters
        self.save_old_params()
        
        # Initialize tracking variables
        total_loss = 0
        total_sample_loss = 0
        total_divergence_loss = 0
        
        for _ in range(self.epochs):
            # Get batch from replay buffer
            states, actions, next_states = replay_pool.sample(self.batch_size)
            
            # Move data to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            next_states = next_states.to(self.device)
            
            # Prepare inputs by concatenating states and actions
            inputs = torch.cat((states, actions.unsqueeze(1)), dim=1)
            
            # Standard precision training
            loss, sample_loss, divergence_loss = self.loss(inputs, next_states, True)
                
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
        
    def get_prediction_with_uncertainty(self, x, n_samples=30):
        """
        Get mean prediction with uncertainty estimates
        Returns: mean prediction, aleatoric uncertainty, epistemic uncertainty, total uncertainty
        """
        x = x.to(self.device)
        
        # Collect prediction samples
        means = []
        stds = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                mean, std = self.forward(x)
                means.append(mean)
                stds.append(std)
                
        # Stack results
        means = torch.stack(means)  # shape: [n_samples, batch_size, output_dim]
        stds = torch.stack(stds)    # shape: [n_samples, batch_size, output_dim]
        
        # Calculate prediction mean and variance
        pred_mean = means.mean(dim=0)  # Average prediction across samples
        
        # Aleatoric uncertainty - average predicted variance
        aleatoric_uncertainty = stds.pow(2).mean(dim=0)
        
        # Epistemic uncertainty - variance of means across samples
        epistemic_uncertainty = means.var(dim=0)
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        return pred_mean, aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty