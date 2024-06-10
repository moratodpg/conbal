from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import wandb
import yaml
import pprint

from models.dnn import SimpleNet, Trainer, MLP_dropout
from datasets.buildings_dataset import Buildings
from active_learning.active_learn import ActiveLearning

def training_loop(config=None):
    with wandb.init(config=config):

        config = wandb.config

        ## Input data
        dataset_th_file = config.dataset

        num_train = 70
        num_test = 3000
        num_epochs = 300
        num_classes = 7

        batch_size = config.batch_size
        learning_rate = config.learning_rate
        hidden_size = config.hidden_size
        layers = config.layers
        dropout_rate = config.dropout

        mode = config.mode # "random"
        number_active_points = config.active_points
        num_active_iter = config.active_iterations
        num_forwards = config.num_forwards
        w_decay = config.weight_decay

        filename = "score_stored"

        # Assuming you have created a Buildings dataset instance named buildings_dataset
        buildings_dataset = torch.load(dataset_th_file)

        # Calculate the length of the dataset
        total_samples = len(buildings_dataset)
        print("Total number of samples in the dataset: ", total_samples)

        num_pool = total_samples - num_train - num_test

        # Load location identifiers
        # with open("datasets/subset_build_6kB_loc_identifier.json", 'r') as f:
        #     loc_identifier = json.load(f)
        # loc_identifier = torch.tensor(loc_identifier)

        # Generate random indices for the downscaled dataset
        # Ensure reproducibility with a fixed seed, if necessary
        # np.random.seed(42)  # Uncomment to make the selection reproducible
        # indices = np.random.choice(len(buildings_dataset), downscaled_size, replace=False)

        ### Split the dataset into training, testing, and pool sets and store the indices
        # Use random_split to split the dataset into training and testing sets
        # train_ds, test_ds, pool_ds = random_split(buildings_dataset, [num_train, num_test, num_pool])

        # # Storing the indices of the datasets
        # subset_build = {}
        # subset_build["train"] = train_ds.indices
        # subset_build["test"] = test_ds.indices
        # subset_build["pool"] = pool_ds.indices

        # with open("datasets/subset_build_6kB_indices.json", 'w') as f:
        #     json.dump(subset_build, f)

        ### Load the indices
        with open("datasets/build_24k_indices_0.json", 'r') as f:
            subset_build = json.load(f)

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

        score_AL = []
        active_learn = ActiveLearning(num_active_points=number_active_points)

        cost_total = 0

        for i in range(num_active_iter):
            print("Active learning iteration: ", i)
            # Create DataLoaders
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            pool_loader = DataLoader(pool_ds, batch_size=len(pool_ds), shuffle=False)


            # Instantiate the classifier
            input_size = 384 # Only considering aerial images
            net = MLP_dropout(input_size, hidden_size, layers, num_classes, dropout_rate)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=w_decay)
            trainer = Trainer(net, num_classes, train_loader, test_loader, criterion, optimizer, num_epochs, patience=400)

            trainer.train()
            score_AL.append(trainer.score)
            wandb.log({"score": trainer.score})
            ## Loop
            idx_pool = pool_ds.indices
            idx_train = train_ds.indices

            trainer.model.load_state_dict(trainer.best_model)

            ## Get active points
            if mode == "active_learning":
                selected_idx_pool = active_learn.get_active_points(trainer.model, num_forwards, buildings_dataset, idx_pool)
            elif mode == "multiple_active_learning":
                selected_idx_pool = active_learn.get_multiple_active_points(trainer.model, num_forwards, buildings_dataset, idx_pool)
            elif mode == "active_learning_cost":
                selected_idx_pool, cost_accum = active_learn.get_active_points_cost(trainer.model, num_forwards, buildings_dataset, idx_pool, loc_identifier)
                cost_total += cost_accum
                wandb.log({"cost_accum": cost_accum})
            else:
                selected_idx_pool = active_learn.get_random_points(idx_pool)
                # selected_idx_pool, cost_accum = active_learn.get_random_points(idx_pool, loc_identifier)
                # cost_total += cost_accum
                # wandb.log({"cost_accum": cost_accum})
            
            ## Updated indices based on selected samples
            idx_pool_ = [idx for idx in idx_pool if idx not in selected_idx_pool]
            idx_train_ = idx_train + selected_idx_pool

            ## Updated subdatasets based on selected samples
            train_ds = torch.utils.data.dataset.Subset(buildings_dataset, idx_train_) 
            pool_ds = torch.utils.data.dataset.Subset(buildings_dataset, idx_pool_) 

            print(len(pool_ds), len(train_ds))

        # Storing results
        filename_id = filename + ".json"
        with open(filename_id, 'w') as f:
            json.dump(score_AL, f)
        
        # Storing the final model
        torch.save(trainer.model.state_dict(), "final_model_stored.pth")
    
if __name__ == "__main__":
    config_file = "config/al.yaml"

    # wandb.login()

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    pprint.pprint(config)

    # Initialize the sweep
    sweep_id = wandb.sweep(config, project="storing_AL")

    # Run the sweep
    wandb.agent(sweep_id, function=training_loop, count=1)

