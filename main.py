from trainer import VIMETrainer
from environments.cartpole import CartPoleEnv
from environments.adventure import AdventureEnv
from policies.dummy import DummyPolicy
from policies.ppo import PPOPolicy
from bnn import BNN
import torch

env = CartPoleEnv()
#env = AdventureEnv()
# TODO: Test PPOPolicy
# policy = DummyPolicy()

# print device
device = "cpu"
print("Using device:", device)
policy = PPOPolicy(env.get_policy_input_dim(), env.get_policy_output_dim(), device=device, hidden_dim=128)

lr = 0.001
bnn = BNN(env.get_model_input_dim(), env.get_model_output_dim(), hidden_dim=128, n_pred=20, batch_size=5, lr=lr, kl_weight = 1 / lr ** 2 * 0.00000, epochs=10, device=device)

trainer = VIMETrainer(
    env=env,
    policy=policy,
    bnn=bnn,
    n_epochs=50,
    n_traj=100,
    output_dir="out_2",
    eta=(1 / bnn.lr ** 2) * 0.00001
)

trainer.train()