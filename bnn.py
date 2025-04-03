import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os

class BNNLayer(nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 std_prior):
        super(BNNLayer, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.std_prior = std_prior
        self.rho_prior = self.std_to_log(self.std_prior)

        self.W = np.random.normal(
            0., std_prior, (self.n_inputs, self.n_outputs))
        self.b = np.zeros((self.n_outputs,), dtype=float)

        W_mu = torch.Tensor(self.n_inputs, self.n_outputs)
        W_mu = nn.init.normal_(W_mu, mean=0., std=1.)
        self.W_mu = nn.Parameter(W_mu)

        W_rho = torch.Tensor(self.n_inputs, self.n_outputs)
        W_rho = nn.init.constant_(W_rho, self.rho_prior)
        self.W_rho = nn.Parameter(W_rho)

        b_mu = torch.Tensor(self.n_outputs, )
        b_mu = nn.init.normal_(b_mu, mean=0., std=1.)
        self.b_mu = nn.Parameter(b_mu)

        b_rho = torch.Tensor(self.n_outputs,)
        b_rho = nn.init.constant_(b_rho, self.rho_prior)
        self.b_rho = nn.Parameter(b_rho)

        self.W_mu_old = torch.Tensor(self.n_inputs, self.n_outputs).detach()
        self.W_rho_old = torch.Tensor(self.n_inputs, self.n_outputs).detach()
        self.b_mu_old = torch.Tensor(self.n_outputs,).detach()
        self.b_rho_old = torch.Tensor(self.n_outputs,).detach()
        
        self.W_mu_init = self.W_mu.clone().detach()
        self.W_rho_init = self.W_rho.clone().detach()
        self.b_mu_init = self.b_mu.clone().detach()
        self.b_rho_init = self.b_rho.clone().detach()

    def forward(self, X, infer=False):
        if infer:
            output = torch.mm(X, self.W_mu) + \
                self.b_mu.expand(X.size()[0], self.n_outputs)
            return output

        W = self.get_W()
        b = self.get_b()
        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_outputs)
        return output

    def get_W(self):
        epsilon = torch.Tensor(self.n_inputs, self.n_outputs)
        epsilon = nn.init.normal_(epsilon, mean=0., std=1.)
        epsilon = Variable(epsilon)
        self.W = self.W_mu + self.log_to_std(self.W_rho) * epsilon
        return self.W

    def get_b(self):
        epsilon = torch.Tensor(self.n_outputs, )
        epsilon = nn.init.normal_(epsilon, mean=0., std=1.)
        epsilon = Variable(epsilon)
        self.b = self.b_mu + self.log_to_std(self.b_rho) * epsilon
        return self.b

    def log_to_std(self, rho):
        return torch.log(1 + torch.exp(rho))

    def std_to_log(self, std):
        return np.log(np.exp(std) - 1)

    def kl_div_new_prior(self):
        kl_div = self.kl_div(
            self.W_mu, self.log_to_std(self.W_rho), self.W_mu_init, self.log_to_std(self.W_rho_init))
        kl_div = kl_div + self.kl_div(
            self.b_mu, self.log_to_std(self.b_rho), self.b_mu_init, self.log_to_std(self.b_rho_init))
        return kl_div

    def kl_div_new_old(self):
        kl_div = self.kl_div(
            self.W_mu, self.log_to_std(self.W_rho), self.W_mu_old, self.log_to_std(self.W_rho_old))
        kl_div = kl_div + self.kl_div(
            self.b_mu, self.log_to_std(self.b_rho), self.b_mu_old, self.log_to_std(self.b_rho_old))
        return kl_div

    def kl_div(self, p_mean, p_std, q_mean, q_std):
        # Fixed: ensuring all inputs are properly handled as tensors
        if not isinstance(q_std, torch.Tensor):
            torch_q_std = torch.Tensor([q_std]).to(p_std.device)
        else:
            torch_q_std = q_std
        numerator = (p_mean - q_mean)**2 + p_std**2 - torch_q_std**2
        denominator = 2 * torch_q_std**2 + 1e-8
        return((numerator / denominator + torch.log(torch_q_std + 1e-8) - torch.log(p_std + 1e-8)).sum())

    def save_old_params(self):
        self.W_mu_old = self.W_mu.clone().detach()
        self.W_rho_old = self.W_rho.clone().detach()
        self.b_mu_old = self.b_mu.clone().detach()
        self.b_rho_old = self.b_rho.clone().detach()

    def reset_to_old_params(self):
        self.W_mu.data = self.W_mu_old.data
        self.W_rho.data = self.W_rho_old.data
        self.b_mu.data = self.b_mu_old.data
        self.b_rho.data = self.b_rho_old.data
        
# TODO: Add learning rate scheduler
class BNN(nn.Module):  # Fixed: Adding nn.Module inheritance
    def __init__(self, input_dim, output_dim, hidden_dim, n_pred=5, num_replay_samples=100, epochs=10, kl_weight=0.01, lr=0.001):
        super(BNN, self).__init__()  # Fixed: Adding super constructor call
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_pred = n_pred
        self.num_replay_samples = num_replay_samples
        self.epochs = epochs
        self.kl_weight = kl_weight
        self.lr = lr
        print("KL weight: ", self.kl_weight)
        print("Learning rate: ", self.lr)
        
        self.layers = nn.ModuleList()
        self.layers.append(BNNLayer(input_dim, hidden_dim, 0.1))
        self.layers.append(nn.ReLU())
        self.layers.append(BNNLayer(hidden_dim, hidden_dim, 0.1))
        self.layers.append(nn.ReLU())
        self.layers.append(BNNLayer(hidden_dim, output_dim, 0.1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, infer=False):  # Fixed: Adding infer parameter
        for layer in self.layers:
            if isinstance(layer, BNNLayer):
                x = layer(x, infer)  # Pass infer parameter to BNNLayer
            else:
                x = layer(x)  # For non-BNNLayer modules like ReLU
        return x
    
    def infer(self, x):  # Added: Inference method
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
        loss = 0
        
        for i in range(self.n_pred):
            output = self.forward(inputs)
            loss += nn.MSELoss()(output, targets)
        
        loss /= self.n_pred
        kl_div = self.kl_div_new_old()
        
        return loss + self.kl_weight * kl_div, loss, kl_div
    
    def loss_last_sample(self, input, target):
        # Fixed: Properly handle inputs that may or may not need reshaping
        if len(input.shape) == 1:  # Single input, needs reshaping
            input = input.unsqueeze(0)  # Add batch dimension
        
        # Calculate average MSE loss over n_pred samples
        loss = 0
        for _ in range(self.n_pred):
            output = self.forward(input)
            # Handle output shape properly
            if len(output.shape) > 1 and output.shape[0] == 1:
                output = output.squeeze(0)  # Remove batch dimension if needed
            loss += nn.MSELoss()(output, target)
        loss /= self.n_pred
        
        return loss
    
    def eval_info_gain(self, inputs, targets):
        info_gain = []
        for i in range(len(inputs)):
            # Save parameters before training on single sample
            self.save_old_params()
            
            # Calculate loss for this sample
            loss_last_sample = self.loss_last_sample(inputs[i], targets[i])
            
            # Perform gradient step
            self.optimizer.zero_grad()
            loss_last_sample.backward()
            self.optimizer.step()
            
            # Calculate KL divergence (information gain)
            info_gain_value = self.kl_div_new_old().item()  # Convert tensor to scalar
            info_gain.append(info_gain_value)
            
            # Restore original parameters
            self.reset_to_old_params()
            
        return torch.tensor(info_gain)
    
    def reset_to_old_params(self):
        for layer in self.layers:
            if isinstance(layer, BNNLayer):
                layer.reset_to_old_params()
    
    def save_old_params(self):
        for layer in self.layers:
            if isinstance(layer, BNNLayer):
                layer.save_old_params()
    
    def update(self, replay_pool):
        # Get sample batch from replay buffer
        self.save_old_params()
        total_loss = 0 
        total_sample_loss = 0
        total_divergence_loss = 0
        for _ in range(self.epochs):
            states, actions, next_states = replay_pool.sample(self.num_replay_samples)
            
            # Prepare inputs and targets
            inputs = torch.cat((states, actions.unsqueeze(1)), dim=1)
            targets = next_states
            
            # Calculate loss and perform optimization step
            loss, sample_loss, divergence_loss = self.loss(inputs, targets)
            total_loss += loss.item()
            total_sample_loss += sample_loss.item()
            total_divergence_loss += divergence_loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Save parameters after update
        
        self.save_old_params()
        
        return total_loss / self.epochs, total_sample_loss / self.epochs, total_divergence_loss / self.epochs

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), path + "/bnn_model.pth")