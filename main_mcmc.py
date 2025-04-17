import datetime
import os
import json
import pprint
import argparse
import random
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
# import wandb
import yaml

from models.mcmc import BayesianNNTrainer
from datasets.buildings_dataset import Buildings
# from active_learning.active_learn import ActiveLearning

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def training_loop(config=None, use_wandb=False, custom_name=None):
    if use_wandb and wandb_available:
        project = config.get("wandb_project", "test_al")
        entity = config.get("wandb_entity", None)  # Optional
        wandb_mode = config.get("wandb_mode", "online")  # Optional
        run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            name=custom_name,
            mode=wandb_mode,
        )
        config = wandb.config

    seed = int(config["seed"])
    set_seed(seed)

    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    num_samples = config["num_samples"]
    parallel_chains = config["parallel_chains"]
    warmup_steps = config["warmup_steps"]
    layers = config["layers"]
    num_forwards = config["num_forwards"]
    number_active_points = config["active_points"]
    num_active_iter = config["active_iterations"]
    budget_total = config["budget_total"]
    al_mode = config["al_mode"]
    mode = config["mode"] 
    save_interval = config["save_interval"]

    dataset_file = config["dataset"]
    buildings_dataset = torch.load("datasets/" + dataset_file + ".pth")
    print("Total number of samples in the dataset: ", len(buildings_dataset))
    # Number of classes
    num_classes = buildings_dataset.output_tensor.max().item() + 1
    # Load coordinates
    coordinates = torch.load("datasets/" + dataset_file + "_coord.pth")
    cost_area = torch.load("datasets/" + dataset_file + "_areacost.pth")["cost_tensor"]
    ### Load the indices
    with open("datasets/" + dataset_file + "_indices_0.json", 'r') as f:
        subset_build = json.load(f)

    start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{custom_name}_{start_time}"
    results_dir = ["results/models/" + file_name, "results/json/" + file_name]
    [os.makedirs(dir, exist_ok=True) for dir in results_dir]

    # Save the updated configuration to a file
    updated_config_filename = os.path.join(results_dir[1], "config.yaml")
    with open(updated_config_filename, 'w') as config_file:
        yaml.dump(dict(config), config_file)

    train_ds = Subset(buildings_dataset, subset_build["train"])
    test_ds = Subset(buildings_dataset, subset_build["test"])
    pool_ds = Subset(buildings_dataset, subset_build["pool"])

    print("Number of samples in the training set: ", len(train_ds))
    print("Number of samples in the testing set: ", len(test_ds))
    print("Number of samples in the pool set: ", len(pool_ds))

    ### To reset the datasets
    # train_ds_0 = train_ds
    # test_ds_0 = test_ds
    # pool_ds_0 = pool_ds

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Instantiate the active learning class
    key_al_module = "active_learning." + mode
    al_class = getattr(importlib.import_module(key_al_module), al_mode)
    active_learn = al_class(number_active_points, budget_total, coordinates, cost_area)

    # Create dictionary with two keys: "accuracy_test" and "cost"
    score_AL = {"accuracy_test": [], "cost": [], "idx_train": [], "idx_pool": [], "idx_test": []}
    score_AL["idx_pool"] = pool_ds.indices
    score_AL["idx_test"] = test_ds.indices
    score_AL["idx_train"] = [train_ds.indices]

    cost_total = 0

    for i in range(num_active_iter):
        print("Active learning iteration: ", i)
        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Instantiate the Bayesian NN trainer
        bnn_trainer = BayesianNNTrainer(
            input_size=384,
            hidden_size=hidden_size,
            num_classes=num_classes,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_samples=int(num_samples/parallel_chains),
            warmup_steps=warmup_steps,
            num_chains=parallel_chains,
        )
        
        # MCMC "training"
        print("Running BayesianNNTrainer MCMC...")
        bnn_trainer.train(train_loader)

        # Evaluate on test set
        test_score = bnn_trainer.evaluate(test_loader, n_samples=num_samples)
        print(f"Test accuracy: {test_score:.4f}")
        score_AL["accuracy_test"].append(test_score)

        ## Loop
        idx_pool = pool_ds.indices
        idx_train = train_ds.indices
            
        selected_idx_pool, cost = active_learn.get_points(bnn_trainer, buildings_dataset, idx_pool, n_samples=num_samples)
        score_AL["cost"].append(cost)
        cost_total += cost

        if use_wandb and wandb_available:
            wandb.log({
            "score": test_score,
            "cost_accum": cost,
            }, step=i)
        
        ## Updated indices based on selected samples
        idx_pool_ = [idx for idx in idx_pool if idx not in selected_idx_pool]
        idx_train_ = idx_train + selected_idx_pool
        score_AL["idx_train"].append(selected_idx_pool)

        ## Updated subdatasets based on selected samples
        train_ds = torch.utils.data.dataset.Subset(buildings_dataset, idx_train_) 
        pool_ds = torch.utils.data.dataset.Subset(buildings_dataset, idx_pool_) 
        print(len(pool_ds), len(train_ds))

        if i % save_interval == 0:
            model_filename = os.path.join(results_dir[0], f"net_{i}.pth")
            torch.save(bnn_trainer.posterior_samples, model_filename)
            print(f"Model saved: {model_filename}")
            model_filename = os.path.join(results_dir[1], "output.json")
            with open(model_filename, 'w') as f:
                json.dump(score_AL, f)
    
    # Storing the final model
    model_filename = os.path.join(results_dir[0], f"net_last.pth")
    torch.save(bnn_trainer.posterior_samples, model_filename)
    print(f"Model saved: {model_filename}")
    model_filename = os.path.join(results_dir[1], "output.json")
    with open(model_filename, 'w') as f:
        json.dump(score_AL, f)
    
    if use_wandb and wandb_available:
        run.finish()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with wandb sweep configuration.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("filename", type=str, nargs='?', default="output", help="Custom name for the output files (optional).")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging.")

    args = parser.parse_args()
    config_file = "config/" + args.config_file + ".yaml"
    # wandb.login()
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if config["seed"] == "None":
        config["seed"] = random.randint(0, 1000000) 
    
    custom_name = args.filename
    pprint.pprint(config)

    training_loop(config=config, use_wandb=args.use_wandb, custom_name=custom_name)

