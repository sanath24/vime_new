from trainer import VIMETrainer
from environments.cartpole import CartPoleEnv
from environments.half_cheetah import HalfCheetahEnv
from environments.mountain_car import MountainCarEnv
from environments.swimmer import SwimmerEnv
from environments.adventure import AdventureEnv
from policies.dummy import DummyPolicy
from policies.ppo import PPOPolicy
from bnn import BNN
import torch
import argparse
import datetime
import os


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train VIME on different environments')
    
    # Add arguments BEFORE parsing
    parser.add_argument('--env', type=str, default='cartpole',
                        choices=['cartpole', 'mountain_car', 'swimmer', 'half_cheetah', 'adventure'],
                        help='Environment to train on')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--n_traj', type=int, default=100,
                        help='Number of trajectories per epoch')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for networks')
    parser.add_argument('--output_dir', type=str, default='out_2',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu/cuda). If None, will use cuda if available')
    
    # Parse arguments AFTER adding them
    args = parser.parse_args()

    if args.env == 'cartpole':
        env = CartPoleEnv()
    elif args.env == 'mountain_car':
        env = MountainCarEnv()
    elif args.env == 'swimmer':
        env = SwimmerEnv()
    elif args.env == 'half_cheetah':
        env = HalfCheetahEnv()
    elif args.env == 'adventure':
        env = AdventureEnv()
    else:
        raise ValueError(f"Unknown environment")
    
    # env = CartPoleEnv()
    #env = AdventureEnv()
    # TODO: Test PPOPolicy
    # policy = DummyPolicy()

    # print device
    device = "cpu"
    if args.device:
        device = args.device
    print("Using device:", device)
    policy = PPOPolicy(env.get_policy_input_dim(), env.get_policy_output_dim(), device=device, hidden_dim=args.hidden_dim)

    lr = 0.001
    bnn = BNN(env.get_model_input_dim(), env.get_model_output_dim(), hidden_dim=args.hidden_dim, n_pred=20, batch_size=5, lr=lr, kl_weight = 1 / lr ** 2 * 0.00001, epochs=10, device=device)

    # timestamp and save every training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.env}_{timestamp}_out")
    os.makedirs(output_dir, exist_ok=True)

    trainer = VIMETrainer(
        env=env,
        policy=policy,
        bnn=bnn,
        n_epochs=args.n_epochs,
        n_traj=args.n_traj,
        output_dir=output_dir,
        eta=(1 / bnn.lr ** 2) * 0.00001
    )

    trainer.train()
    
    
    
