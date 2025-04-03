from trainer import VIMETrainer
from environments.cartpole import CartPoleEnv
from policies.dummy import DummyPolicy
from policies.ppo import PPOPolicy
from bnn import BNN
import torch

env = CartPoleEnv()
# TODO: Test PPOPolicy
# policy = DummyPolicy()

# print device
device = "cpu"
print("Using device:", device)
policy = PPOPolicy(env.get_policy_input_dim(), env.get_policy_output_dim(), device=device)
lr = 0.001
bnn = BNN(env.get_model_input_dim(), env.get_model_output_dim(), hidden_dim=32, n_pred=10, batch_size=5, lr=lr, kl_weight = 1 / lr ** 2 * 0.00001, epochs=10, device=device)

trainer = VIMETrainer(
    env=env,
    policy=policy,
    bnn=bnn,
    n_epochs=50,
    n_traj=100,
    output_dir="out_0",
    eta=(1 / bnn.lr ** 2) * 0.0001
)

trainer.train()