import subprocess
import os

# remove adventure for now
environments = ['cartpole', 'mountain_car', 'half_cheetah', 'walker_2d']
schedulers = ['default', 'linear', 'regularization', 'warmup']
schedulers = reversed(schedulers)
environments = reversed(environments)
num_epochs = [100, 50]
n_traj = 100
hidden_dim = 128

for num in num_epochs:
    for scheduler in schedulers:
        for env in environments:            
            # Output directory name based on scheduler, number of epochs, and environment
            output_dir = f"experiments/{scheduler}_{env}_{num}"
            os.makedirs(output_dir, exist_ok=True)
            print(f"{output_dir}")

            command = [
                "python3", "main.py",
                "--env", env,
                "--n_epochs", str(num),
                "--n_traj", str(n_traj),
                "--hidden_dim", str(hidden_dim),
                "--output_dir", output_dir,
                "--device", "cpu",
                "--scheduler", scheduler,
                "--sparsity", "mean"  # Fixing sparsity to 'mean'
            ]
            subprocess.run(command)
            
    
            

            # # log each run to txt file
            # log_file = f"{output_dir}_log.txt"
            # with open(log_file, "w") as log:
            #     subprocess.run(command, stdout=log, stderr=log)