import subprocess

environments = ['cartpole', 'mountain_car', 'swimmer', 'half_cheetah', 'adventure', 'walker_2d']
schedulers = ['linear', 'regularization', 'warmup']
num_epochs = [50,100]
n_traj = 100
hidden_dim = 128

for num in num_epochs:
    for env in environments:
        for scheduler in schedulers:
            # Output directory name based on scheduler, number of epochs, and environment
            output_dir = f"{scheduler}_{env}_{num}"
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