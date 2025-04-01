import torch
from replay_buffer import ReplayBuffer
from trajectory import Trajectory
from environment import Environment
from bnn import BNN
from policy import Policy
from tqdm import tqdm

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
        for _ in tqdm(range(self.n_epochs)):
            trajectories = self.sample_trajectories()
            states = []
            actions = []
            rewards = []
            next_states = []
            log_probs = []
            dones = []
            
            for traj in trajectories:
                traj_states, traj_actions, traj_next_states, traj_rewards, traj_log_probs = traj.get_inputs_and_targets()
                traj_dones = torch.zeros(traj_states.shape[0])
                # set the last element of traj_dones to 1
                traj_dones[-1] = 1
                info_gain = self.bnn.eval_info_gain(traj_states, traj_actions, traj_next_states)
                new_rewards = traj_rewards + info_gain
                rewards.append(new_rewards)
                states.append(traj_states)
                actions.append(traj_actions)
                next_states.append(traj_next_states)
                log_probs.append(traj_log_probs)
                dones.append(traj_dones)
            
            self.bnn.update(self.replay_pool)
            self.policy.update(states, actions, rewards, next_states, log_probs, dones)
                
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