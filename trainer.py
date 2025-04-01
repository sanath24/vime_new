import torch
from replay_buffer import ReplayBuffer
from trajectory import Trajectory
from environment import Environment
from bnn import BNN
from policy import Policy
from tqdm import tqdm
import numpy as np

class VIMETrainer():
    def __init__(self, env: Environment, policy: Policy, bnn: BNN, n_epochs, n_traj):
        self.n_epochs = n_epochs
        self.n_traj = n_traj
        self.env = env
        self.policy = policy
        self.bnn = bnn
        self.replay_pool = ReplayBuffer()
    
    # TODO: Log results
    def train(self):
        for _ in range(self.n_epochs):
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
                info_gain = self.bnn.eval_info_gain(traj_states, traj_actions, traj_next_states)
                new_rewards = traj_rewards + info_gain
                old_rewards.append(traj_rewards)
                info_gains.append(info_gain)
                rewards.append(new_rewards)
                states.append(traj_states)
                actions.append(traj_actions)
                next_states.append(traj_next_states)
                log_probs.append(traj_log_probs)
                dones.append(traj_dones)
            
            self.bnn.update(self.replay_pool)
            self.policy.update(states, actions, rewards, next_states, log_probs, dones)
            self.replay_pool.clear()
            self.log_results(rewards, old_rewards, info_gains)
                
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

    def log_results(self, rewards, old_rewards, info_gains):
        rewards = np.mean([torch.sum(r) for r in rewards])
        old_rewards = np.mean([torch.sum(r) for r in old_rewards])
        info_gains = np.mean([torch.sum(r) for r in info_gains])
    
        
        print(f"Rewards: {rewards}, Old Rewards: {old_rewards}, Info Gains: {info_gains}")