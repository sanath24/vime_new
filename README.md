## About

This is our implementation of VIME (Variational Information Maximizing Exploration), a reinforcement learning method introduced by OpenAI in 2016. Paper Link - https://arxiv.org/abs/1605.09674

We implemented the paper with PyTorch and introduced more optimized BNN update strategy. We also implement a scheduler for the hyperparameter controlling exploration-exploitation tradeoff. 

## Setup

To setup create a virtual environment and install dependencies with pip install -r requirements.txt

To run all experiments, switch to the eta-scheduler branch and run python3 run_experiments.py
