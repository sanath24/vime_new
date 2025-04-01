from trainer import VIMETrainer
from environments.cartpole import CartPoleEnv
from policies.dummy import DummyPolicy
from policies.ppo import PPOPolicy
from bnn import BNN

env = CartPoleEnv()
# TODO: Test PPOPolicy
# policy = DummyPolicy()
policy = PPOPolicy(env.get_state_dim(), env.get_action_dim(),)
bnn = BNN()

trainer = VIMETrainer(
    env=env,
    policy=policy,
    bnn=bnn,
    n_epochs=100,
    n_traj=10
)

trainer.train()