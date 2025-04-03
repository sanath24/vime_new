import torch
from replay_buffer import ReplayBuffer
from trajectory import Trajectory
from environment import Environment
from bnn import BNN
from policy import Policy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

class VIMETrainer():
    def __init__(self, env: Environment, policy: Policy, bnn: BNN, n_epochs, n_traj, eta=0, output_dir=None):
        self.n_epochs = n_epochs
        self.n_traj = n_traj
        self.env = env
        self.policy = policy
        self.bnn = bnn
        self.replay_pool = ReplayBuffer()
        self.eta = eta
        self.results = []
        self.output_dir = output_dir
        print("ETA: ", eta)
    
    # TODO: Log results
    def train(self):
        for i in range(self.n_epochs):
            trajectories = self.sample_trajectories()
            states = []
            actions = []
            old_rewards = []
            info_gains = []
            rewards = []
            next_states = []
            log_probs = []
            dones = []
            
            for traj in tqdm(trajectories):
                traj_states, traj_actions, traj_next_states, traj_rewards, traj_log_probs = traj.get_inputs_and_targets()
                traj_dones = torch.zeros(traj_states.shape[0])
                # set the last element of traj_dones to 1
                traj_dones[-1] = 1
                info_gain = self.bnn.eval_info_gain(torch.cat((traj_states, traj_actions.unsqueeze(1)), dim=1), traj_next_states)
                traj_rewards = traj_rewards.to(self.bnn.device)
                new_rewards = traj_rewards + self.eta * info_gain
                old_rewards.append(traj_rewards)
                info_gains.append(info_gain)
                rewards.append(new_rewards)
                states.append(traj_states)
                actions.append(traj_actions)
                next_states.append(traj_next_states)
                log_probs.append(traj_log_probs)
                dones.append(traj_dones)
            
            bnn_loss, sample_loss, divergence_loss = self.bnn.update(self.replay_pool)
            self.policy.update(states, actions, rewards, next_states, log_probs, dones)
            self.replay_pool.clear()
            self.log_results(i, rewards, old_rewards, info_gains, bnn_loss, sample_loss, divergence_loss)
            
        self.policy.save_model(self.output_dir)
        self.bnn.save_model(self.output_dir)
        self.save_results()
                
    def sample_trajectories(self) -> list[Trajectory]:
        result = []
        for n in range(self.n_traj):
            trajectory = self.rollout()
            result.append(trajectory)
        
        return result
    
    def rollout(self, add_to_replay=True):
        self.env.reset()
        trajectory = Trajectory(self.env.get_start_state())
        terminal = False
        while not terminal:
            current_state = trajectory.get_current_state()
            next_action, log_prob = self.policy.get_action(current_state)
            next_state, reward, terminal = self.env.step(next_action)
            trajectory.add_step(next_action, next_state, reward, log_prob)
            if add_to_replay:
                self.replay_pool.add(current_state, next_action, next_state)
        
        return trajectory

    def log_results(self, epoch, rewards, old_rewards, info_gains, bnn_loss, sample_loss, divergence_loss):
        rewards = np.mean([torch.sum(r).cpu() for r in rewards])
        old_rewards = np.mean([torch.sum(r).cpu() for r in old_rewards])
        # concatenate info gains
        info_gains = torch.cat(info_gains)
        info_gains = torch.mean(info_gains)
    
        
        print(f"Epoch: {epoch}, Rewards: {rewards}, Old Rewards: {old_rewards}, Info Gains: {info_gains}, BNN Loss: {bnn_loss}, Sample Loss: {sample_loss}, Divergence Loss: {divergence_loss}")
    
        self.results.append((rewards, old_rewards, info_gains, bnn_loss, sample_loss, divergence_loss))
        
    def save_results(self):
        # Convert results to a NumPy array
        results = np.array(self.results)
        
        # Extract different losses and metrics
        rewards = results[:, 0]
        old_rewards = results[:, 1]
        info_gains = results[:, 2]
        bnn_loss = results[:, 3]
        sample_loss = results[:, 4]
        divergence_loss = results[:, 5]
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, label="Rewards")
        plt.legend()
        plt.title("Total Rewards Over Time")
        # Save plot
        output_path = os.path.join(self.output_dir, "results.png")
        plt.savefig(output_path)
        plt.close()  # Close the plot to free memory
        plt.figure(figsize=(10, 6))
        plt.plot(old_rewards, label="Old Rewards")
        plt.legend()
        plt.title("Actual Rewards Over Time")
        # Save plot
        output_path = os.path.join(self.output_dir, "actual_rewards.png")
        plt.savefig(output_path)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(info_gains, label="Info Gains")
        plt.legend()
        plt.title("Info Gains Over Time")
        # Save plot
        output_path = os.path.join(self.output_dir, "info_gains.png")
        plt.savefig(output_path)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(bnn_loss, label="BNN Loss")
        plt.legend()
        plt.title("BNN Loss Over Time")
        # Save plot
        output_path = os.path.join(self.output_dir, "bnn_loss.png")
        plt.savefig(output_path)
        
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(sample_loss, label="Sample Loss")
        plt.legend()
        plt.title("Sample Loss Over Time")
        # Save plot
        output_path = os.path.join(self.output_dir, "sample_loss.png")
        plt.savefig(output_path)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(divergence_loss, label="Divergence Loss")
        plt.legend()
        plt.title("Divergence Loss Over Time")
        # Save plot
        output_path = os.path.join(self.output_dir, "divergence_loss.png")
        plt.savefig(output_path)
        plt.close()
        

            