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
import datetime
from scipy.stats import entropy, kurtosis
import pandas as pd


class VIMETrainer():
    def __init__(self, env: Environment, policy: Policy, bnn: BNN, n_epochs, n_traj, eta=0, output_dir=None, sparsity_est="mean", scheduler="linear"):
        self.n_epochs = n_epochs
        self.n_traj = n_traj
        self.env = env
        self.policy = policy
        self.bnn = bnn
        self.replay_pool = ReplayBuffer()
        self.eta = eta
        self.sparsity_est = sparsity_est
        self.scheduler=scheduler
        print(f"Trainer initialized with eta: {self.eta}")
        self.results = []
        self.output_dir = output_dir
        os.makedirs("logs",exist_ok=True)
        # timestamp and save every training run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("logs", f"{timestamp}_logs")
        os.makedirs(output_dir, exist_ok=True)
        self.save_dir = output_dir
        print("ETA: ", eta)
        self.etas = [self.eta]
    

    def sparsity_nonzero_ratio(self, rewards):
        """
        Fraction of steps with nonzero reward.
        Lower ratio ⇒ more sparse.
        """
        rewards = np.array(rewards)
        if rewards.size == 0:
            return 0.0
        return 1.0 - (np.count_nonzero(rewards) / rewards.size)


    def sparsity_unique_ratio(self, rewards):
        """
        Ratio of unique reward values to total steps.
        Lower ratio ⇒ more clustering (i.e., sparser signal).
        """
        rewards = np.array(rewards)
        if rewards.size == 0:
            return 0.0
        return len(np.unique(rewards)) / rewards.size


    def sparsity_histogram_entropy(self, rewards, bins=10):
        """
        Shannon entropy of the reward histogram.
        Lower entropy ⇒ mass concentrated in fewer bins ⇒ sparse.
        """
        rewards = np.array(rewards)
        hist, _ = np.histogram(rewards, bins=bins, density=True)
        # add a small epsilon so log doesn’t blow up on zeros
        hist = hist + 1e-12
        return entropy(hist)


    def sparsity_renyi_entropy(self, rewards, alpha=2, bins=10):
        """
        Renyi‐α entropy of the reward histogram.
        For α>1, more sensitive to high‐probability bins.
        """
        rewards = np.array(rewards)
        hist, _ = np.histogram(rewards, bins=bins, density=True)
        hist = hist + 1e-12
        return 1.0 / (1 - alpha) * np.log((hist**alpha).sum())


    def sparsity_gini_coefficient(self, rewards):
        """
        Gini coefficient on absolute reward magnitudes.
        Higher Gini ⇒ more inequality in the distribution ⇒ sparser.
        """
        vals = np.abs(np.array(rewards)).flatten()
        if vals.size == 0:
            return 0.0
        sorted_vals = np.sort(vals)
        n = vals.size
        index = np.arange(1, n+1)
        return (np.sum((2*index - n - 1) * sorted_vals) / (n * np.sum(sorted_vals) + 1e-12))


    def sparsity_iqr_ratio(self, rewards):
        """
        Interquartile range (IQR) divided by total range.
        Lower IQR / range ⇒ rewards are more peaked ⇒ sparser.
        """
        rewards = np.array(rewards)
        if rewards.size == 0:
            return 0.0
        q75, q25 = np.percentile(rewards, [75 ,25])
        iqr = q75 - q25
        data_range = rewards.max() - rewards.min() + 1e-12
        return 1.0 - (iqr / data_range)


    def sparsity_coefficient_of_variation(self, rewards):
        """
        Coefficient of variation: std / |mean|.
        Lower CV ⇒ less variation relative to mean ⇒ sparser.
        """
        rewards = np.array(rewards)
        if rewards.size == 0:
            return 0.0
        mu = rewards.mean()
        sigma = rewards.std()
        return sigma / (abs(mu) + 1e-12)


    def sparsity_kurtosis(self, rewards):
        """
        Excess kurtosis of the reward distribution.
        Higher kurtosis ⇒ heavier tails and sharper peak ⇒ sparser.
        """
        rewards = np.array(rewards)
        if rewards.size < 4:
            return 0.0
        return kurtosis(rewards, fisher=True)


    def sparsity_autocorrelation(self, rewards, lag=1):
        """
        1-step autocorrelation. 
        High autocorrelation ⇒ rewards change slowly ⇒ sparser signal.
        """
        rewards = np.array(rewards)
        if rewards.size <= lag:
            return 0.0
        r = rewards
        return np.corrcoef(r[:-lag], r[lag:])[0,1]


    # -------------------------------------------------------------------------
    # Standard‐deviation based approach
    # -------------------------------------------------------------------------
    def sparsity_std_measure(self, rewards):
        """
        Use the raw standard deviation as a measure of variation.
        Low std ⇒ rewards are clustered (i.e. sparse).
        
        Returns:
            float: The standard deviation.
        """
        rewards = np.array(rewards)
        if rewards.size == 0:
            return 0.0
        return rewards.std()

    """
    Estimating reward sparsity by counting the ratio of nonzeros/total steps
    """
    def estimate_sparsity_nonzero(self, rewards):
        rewards = np.array(rewards)
        total_steps = rewards.size
        if total_steps == 0:
            return 0.0
        nonzero_count = np.count_nonzero(rewards)
        return nonzero_count / total_steps

    """
    creates a histogram of observed rewards and calculates the Shannon entropy, lower entropy means less bins = more sparse
    """
    def estimate_reward_entropy(self, rewards, bins=10):
        rewards = np.array(rewards)
        hist, _ = np.histogram(rewards, bins=bins, density=True)
        
        hist = hist[hist > 0] # Remove zero-probability entries for stability
        return entropy(hist)
    
    """
    Estimate sparsity based on the number of rewards that are smaller than the mean
    """
    def estimate_sparsity_mean(self, rewards):
        
        if rewards.size == 0:
            return 0.0
        mean_rewards = np.mean(rewards)
        return (np.count_nonzero(rewards > mean_rewards)/rewards.size) * 100
        
    # """

    # Objective is to boost eta when rewards are sparse, decrease eta otherwise


    # """

    # def eta_scheduler_linear(self, current_eta, sparsity_measure, threshold=0.5, 
    #                        increase_rate=0.05, decrease_rate=0.05, 
    #                        eta_min=1.0, eta_max=1000.0):
        
    #     if sparsity_measure



    """
    Boost eta when sparsity below threshold, else decrease
    """
    def eta_scheduler_linear(self, current_eta, sparsity_measure, threshold=0.5, 
                           increase_rate=0.05, decrease_rate=0.05, 
                           eta_min=1.0, eta_max=1000.0):
        
        # For the nonzero ratio, a lower value indicates higher sparsity.
        if sparsity_measure < threshold:
            # Increase eta when rewards are sparse
            new_eta = current_eta * (1 + increase_rate)
        else:
            # Decrease eta when rewards are dense
            new_eta = current_eta * (1 - decrease_rate)
        
        new_eta = np.clip(new_eta, eta_min, eta_max)
        return new_eta
    
    def eta_scheduler_warmup(self, current_eta, epoch, sparsity_measure, threshold=0.5, 
                           increase_rate=0.05, decrease_rate=0.05, 
                           eta_min=1.0, eta_max=1000.0, warmup_epochs=5):
        
        # For the nonzero ratio, a lower value indicates higher sparsity.
        if epoch < warmup_epochs:
            # warmup_eta = eta_min + (eta_max - eta_min) * (epoch / warmup_epochs)
            # return np.clip(warmup_eta, eta_min, eta_max)
            return current_eta
        if sparsity_measure < threshold:
            # Increase eta when rewards are sparse
            # new_eta = current_eta * (1 + increase_rate)
            exploration_factor = 1 + increase_rate
        else:
            # Decrease eta when rewards are dense
            # new_eta = current_eta * (1 - decrease_rate)
            exploration_factor = 1 - decrease_rate

        new_eta = current_eta * exploration_factor
        return np.clip(new_eta, eta_min, eta_max)


    """
    Adjust eta based on a regularization-based approach leveraging BNN's KL div.
    """
    def eta_scheduler_regularized(self, current_eta, average_kl_div, kl_threshold=0.05, 
                                 adjust_rate=0.1, eta_min=1.0, eta_max=1000.0):
        if average_kl_div < kl_threshold:
            # Low KL means overconfient model so boost exploration by increasing eta
            new_eta = current_eta * (1 + adjust_rate)
        else:
            # Sufficient KL means reduce eta to favor exploitation
            new_eta = current_eta * (1 - adjust_rate)
        
        new_eta = np.clip(new_eta, eta_min, eta_max)
        return new_eta

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
            total_bnn_loss = 0
            total_sample_loss = 0
            total_divergence_loss = 0
            self.bnn.reset_kl_div_hist()
            
            for traj in tqdm(trajectories):
                traj_states, traj_actions, traj_next_states, traj_rewards, traj_log_probs = traj.get_inputs_and_targets()
                traj_dones = torch.zeros(traj_states.shape[0])
                # set the last element of traj_dones to 1
                traj_dones[-1] = 1
                # info_gain = self.bnn.eval_info_gain(torch.cat((traj_states, traj_actions.unsqueeze(1)), dim=1), traj_next_states)
                info_gain, bnn_loss, sample_loss, divergence_loss = self.bnn.eval_info_gain_and_update(torch.cat((traj_states, traj_actions.unsqueeze(1)), dim=1), traj_next_states)
                
                total_bnn_loss += bnn_loss
                total_sample_loss += sample_loss
                total_divergence_loss += divergence_loss
                
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
            
            # bnn_loss, sample_loss, divergence_loss = self.bnn.update(self.replay_pool)
            self.policy.update(states, actions, rewards, next_states, log_probs, dones)
            self.replay_pool.clear()
            self.log_results(i, rewards, old_rewards, info_gains, total_bnn_loss / len(trajectories), total_sample_loss / len(trajectories), total_divergence_loss / len(trajectories))

            
            
            all_old_rewards = np.concatenate([r.cpu().numpy() for r in old_rewards])
            # reward_mean = np.mean(all_old_rewards)

            # Option 1 - nonzero based sparsity
            if self.sparsity_est == "nonzero":
                sparsity_val = self.estimate_sparsity_nonzero(all_old_rewards)
                print(f"Epoch {i} - Reward Ratio: {sparsity_val:.4f}")
            # Option 2 - entropy based sparsity
            elif self.sparsity_est == "entropy":
                sparsity_val = self.estimate_reward_entropy(all_old_rewards)
                print(f"Epoch {i} - Reward Entropy: {sparsity_val:.4f}")
            # Option 3 - estimate sparsity based on ratio of rewards above mean 
            elif self.sparsity_est == "mean":
                sparsity_val = self.estimate_sparsity_mean(all_old_rewards)
                print(f"Epoch {i} - Reward Sparsity: {sparsity_val:.4f}")
            else:
                sparsity_val = sparsity_val = self.estimate_sparsity_mean(all_old_rewards)
                print(f"Defaulted to mean sparisty - Epoch {i} - Reward Sparsity: {sparsity_val:.4f}")

            # Approach 1: Linear Approach
            if self.scheduler == 'linear':
                self.eta = self.eta_scheduler_linear(self.eta, sparsity_val, threshold=0.5, 
                                                    increase_rate=0.05, decrease_rate=0.05,
                                                    eta_min=1.0, eta_max=1000.0)
                self.etas.append(self.eta)
            # Approach 2: Regularization Approach
            elif self.scheduler == 'regularization':
                avg_divergence = total_divergence_loss / len(trajectories)
                self.eta = self.eta_scheduler_regularized(self.eta, avg_divergence, kl_threshold=0.05, adjust_rate=0.1,
                                                        eta_min=1.0, eta_max=1000.0)
            # Approach 3: Warmup Linear 
            elif self.scheduler == 'warmup':
                self.eta = self.eta_scheduler_warmup(self.eta, i, sparsity_val, threshold=0.5, 
                                                    increase_rate=0.05, decrease_rate=0.05,
                                                    eta_min=1.0, eta_max=1000.0)
            # logging for later visualization
            self.etas.append(self.eta)

            print(f"Epoch {i} - Updated eta: {self.eta:.4f}")

        self.policy.save_model(self.output_dir)
        self.bnn.save_model(self.output_dir)
        self.save_results()
                
    def sample_trajectories(self) -> list[Trajectory]:
        result = []
        for n in tqdm(range(self.n_traj)):
            trajectory = self.rollout()
            result.append(trajectory)
        
        return result
    
    def rollout(self, add_to_replay=True):
        self.env.reset()
        trajectory = Trajectory(self.env.get_start_state())
        terminal = False
        i = 0
        rand = np.random.rand()
        if rand < 0.1:
            self.env.render()
        else:
            self.env.stop_render()
        while not terminal and i < 2000:
            current_state = trajectory.get_current_state()
            i += 1
            
            next_action, log_prob = self.policy.get_action(current_state)
            # print(f"Raw action type: {type(next_action)}, shape: {next_action.shape if hasattr(next_action, 'shape') else 'no shape'}")
            
            # # Handle both discrete and continuous actions
            # if isinstance(next_action, torch.Tensor):
            #     next_action = next_action.detach().cpu().numpy()
            #     # print(f"After tensor conversion: {next_action.shape}")
            
            # # For continuous actions (like HalfCheetah)
            # if isinstance(next_action, np.ndarray):
            #     if next_action.ndim == 1:
            #         next_action = next_action.reshape(-1)  # Ensure it's a 1D array
            #     print(f"Final action shape: {next_action.shape}")
            # print(f"{type(next_action)}, {next_action}")
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
        etas = self.etas
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # save results
        df = pd.DataFrame({
            'Epoch': np.arange(len(rewards)),
            'Rewards': rewards,
            'Old Rewards': old_rewards,
            'Info Gains': info_gains,
            'BNN Loss': bnn_loss,
            'Sample Loss': sample_loss,
            'Divergence Loss': divergence_loss,
            'Eta': etas
        })
        df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
            
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

        plt.figure(figsize=(10, 6))
        plt.plot(bnn_loss, label="BNN Loss")
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.legend()
        plt.title("BNN Loss Over Time (Log Scale)")
        # Save plot
        output_path = os.path.join(self.output_dir, "bnn_loss_log.png")
        plt.savefig(output_path)
        plt.close()

        
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

        plt.figure(figsize=(10, 6))
        plt.plot(etas, label="Eta")
        plt.legend()
        plt.title("Eta Over Time")
        output_path = os.path.join(self.output_dir, "etas.png")
        plt.savefig(output_path)
        plt.close()
        

            