from trainer import VIMETrainer
from environments.cartpole import CartPoleEnv
from policies.dummy import DummyPolicy
from policies.ppo import PPOPolicy
from bnn import BNN

env = CartPoleEnv()
# TODO: Test PPOPolicy
# policy = DummyPolicy()


policy = PPOPolicy(env.get_policy_input_dim(), env.get_policy_output_dim())
lr = 0.001
bnn = BNN(env.get_model_input_dim(), env.get_model_output_dim(), hidden_dim=32, n_pred=20, num_replay_samples=100, lr=lr, kl_weight = 1 / lr ** 2 * 0.000001, epochs=10)

trainer = VIMETrainer(
    env=env,
    policy=policy,
    bnn=bnn,
    n_epochs=100,
    n_traj=100,
    output_dir="out_0",
    eta=(1 / bnn.lr ** 2) * 0.000005
)

trainer.train()