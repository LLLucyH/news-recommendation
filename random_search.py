import random
import os
import time

random.seed(0)

search_space = {
    "max_title_len": [10, 20, 30, 50],
    "max_hist_len": [10, 20, 30, 50],
    "max_ent_len": [3, 5, 8, 10],
    "hidden_dim": [64, 128, 256],
    "batch_size": [128, 256, 512],
    "lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    "neg_samples": [2, 4, 8],
    "dropout": [0.1, 0.2, 0.3, 0.5]
}

NUM_EXPERIMENTS = 10 


if not os.path.exists("logs"):
    os.makedirs("logs")

for i in range(NUM_EXPERIMENTS):
    t_len = random.choice(search_space["max_title_len"])
    h_len = random.choice(search_space["max_hist_len"])
    e_len = random.choice(search_space["max_ent_len"])
    h_dim = random.choice(search_space["hidden_dim"])
    bs = random.choice(search_space["batch_size"])
    lr = random.choice(search_space["lr"])
    neg = random.choice(search_space["neg_samples"])
    drp = random.choice(search_space["dropout"])

    cmd = f"sbatch train.sh {t_len} {h_len} {e_len} {h_dim} {bs} {lr} {neg} {drp}"
    
    print(f"Launching Job {i+1}: {cmd}")
    os.system(cmd)
    time.sleep(1)