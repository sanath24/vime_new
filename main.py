from trainer import VIMETrainer
from environments.cartpole import CartPoleEnv
from policies.dummy import DummyPolicy
from bnn import BNN

env = CartPoleEnv()
# TODO: Test PPOPolicy
policy = DummyPolicy()
bnn = BNN()

trainer = VIMETrainer(
    env=env,
    policy=policy,
    bnn=bnn,
    n_epochs=100,
    n_traj=10
)

trainer.train()