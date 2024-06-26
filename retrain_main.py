import datetime
import os
import json
import pprint
import argparse
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import wandb
import yaml

from models.dnn import MLP_dropout
from datasets.buildings_dataset import Buildings
from active_learning.active_learn import ActiveLearning

# Define a class for training and testing the classifier
class Retrainer:
    def __init__(self, model, num_classes, test_loader, criterion, optimizer, num_epochs, patience=30):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_classes = num_classes
        self.score = 0
        self.best_val_score = float('-inf')
        self.epochs_without_improvement = 0
        self.best_model = None

    def train(self, train_loader):
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            for inputs, labels, _ in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # print(loss.item())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Evaluate the model on the test set to calculate validation loss
            val_score = self.evaluate()

            # Early stopping check
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.epochs_without_improvement = 0
                self.best_model = copy.deepcopy(self.model.state_dict())
                # print(f"Validation score is now: {val_score:.4f}")
            else:
                self.epochs_without_improvement += 1
                #print(f"Validation loss did not decrease, count: {self.epochs_without_improvement}")
                if self.epochs_without_improvement >= self.patience:
                    print("Early stopping triggered", ", Epoch: " ,epoch + 1)
                    self.test()
                    break

            if epoch == (self.num_epochs - 1):
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {running_loss / len(train_loader)}")
                # Print average loss for the epoch
                # print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {running_loss / len(self.train_loader)}")

                # Test the model after each epoch
                self.test()

    def compute_scores(self):
        correct = 0
        total = 0
        TP, FP, FN, TN = [0] * self.num_classes, [0] * self.num_classes, [0] * self.num_classes, [0] * self.num_classes
        with torch.no_grad():
            for inputs, labels, _ in self.test_loader:
                outputs = self.model(inputs)
                _, predicted_labels = torch.max(outputs, 1) 
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()
                
                # Update TP, FP, TN, FN
                for i in range(self.num_classes):
                    TP[i] += ((predicted_labels == i) & (labels == i)).sum().item()
                    FP[i] += ((predicted_labels == i) & (labels != i)).sum().item()
                    FN[i] += ((predicted_labels != i) & (labels == i)).sum().item()
                    TN[i] += ((predicted_labels != i) & (labels != i)).sum().item()
        
        # Compute precision, recall, and accuracy
        precision = [TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0 for i in range(self.num_classes)]
        recall = [TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0 for i in range(self.num_classes)]
        f1_score = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0 for i in range(self.num_classes)]

        # To compute the average across classes
        avg_precision = sum(precision) / self.num_classes
        avg_recall = sum(recall) / self.num_classes
        avg_f1_score = sum(f1_score) / self.num_classes

        # Overall accuracy
        accuracy = correct / total if total > 0 else 0
        scores = {
            "accuracy": accuracy,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1_score": avg_f1_score
        }
        return scores


    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        scores = self.compute_scores()
        return scores["accuracy"]

    def test(self):
        self.model.load_state_dict(self.best_model)
        self.model.eval()  # Set the model to evaluation mode
        scores = self.compute_scores()
        self.score = scores["accuracy"]
        # Print the computed metrics
        print(f"Accuracy on the test set: {scores['accuracy']}")
        print(f"Average Precision on the test set: {scores['avg_precision']}")
        print(f"Average Recall on the test set: {scores['avg_recall']}")
        print(f"Average F1 Score on the test set: {scores['avg_f1_score']}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def training_loop(config=None):
    with wandb.init(config=config, mode="online", name=custom_name):
        config = wandb.config

        seed = int(config.seed)
        set_seed(seed)

        num_epochs = config.num_epochs
        batch_size = config.batch_size
        learning_rate = config.learning_rate
        w_decay = config.weight_decay
        hidden_size = config.hidden_size
        layers = config.layers
        num_forwards = config.num_forwards
        dropout_rate = config.dropout
        number_active_points = config.active_points
        num_active_iter = config.active_iterations
        mode = config.mode # "random"
        independent_mode = config.independent_mode
        save_interval = config.save_interval

        dataset_file = config.dataset
        buildings_dataset = torch.load("datasets/" + dataset_file + ".pth")
        print("Total number of samples in the dataset: ", len(buildings_dataset))
        # Number of classes
        num_classes = buildings_dataset.output_tensor.max().item() + 1
        # Load coordinates
        coordinates = torch.load("datasets/" + dataset_file + "_coord.pth")
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
        active_learn = ActiveLearning(num_active_points=number_active_points)

        # Instantiate the classifier
        input_size = 384 # Only considering aerial images
        net = MLP_dropout(input_size, hidden_size, layers, num_classes, dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=w_decay)

        trainer = Retrainer(net, num_classes, test_loader, criterion, optimizer, num_epochs, patience=400)

        # To allocate: train_loader,

        # Create dictionary with two keys: "accuracy_test" and "cost"
        score_AL = {"accuracy_test": [], "cost": [], "idx_train": [], "idx_pool": [], "idx_test": []}
        cost_total = 0

        for i in range(num_active_iter):
            print("Active learning iteration: ", i)
            # Create DataLoaders
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            trainer.train(train_loader)
            score_AL["accuracy_test"].append(trainer.score)
            wandb.log({"score": trainer.score})
            ## Loop
            idx_pool = pool_ds.indices
            idx_train = train_ds.indices
            idx_test = test_ds.indices

            trainer.model.load_state_dict(trainer.best_model)

            selected_idx_pool, cost = active_learn.get_points(mode, independent_mode, trainer.model, num_forwards, buildings_dataset, idx_pool, coordinates)
            score_AL["cost"].append(cost)
            wandb.log({"cost_accum": cost})
            cost_total += cost
            
            ## Updated indices based on selected samples
            idx_pool_ = [idx for idx in idx_pool if idx not in selected_idx_pool]
            idx_train_ = idx_train + selected_idx_pool
            score_AL["idx_train"] = idx_train_
            score_AL["idx_pool"] = idx_pool_
            score_AL["idx_test"] = idx_test

            ## Updated subdatasets based on selected samples
            train_ds = torch.utils.data.dataset.Subset(buildings_dataset, idx_train_) 
            pool_ds = torch.utils.data.dataset.Subset(buildings_dataset, idx_pool_) 
            print(len(pool_ds), len(train_ds))

            if i % save_interval == 0:
                model_filename = os.path.join(results_dir[0], f"net_{i}.pth")
                torch.save(trainer.model.state_dict(), model_filename)
                print(f"Model saved: {model_filename}")
                model_filename = os.path.join(results_dir[1], "output.json")
                with open(model_filename, 'w') as f:
                    json.dump(score_AL, f)
        
        # Storing the final model
        model_filename = os.path.join(results_dir[0], f"net_last.pth")
        torch.save(trainer.model.state_dict(), model_filename)
        print(f"Model saved: {model_filename}")
        model_filename = os.path.join(results_dir[1], "output.json")
        with open(model_filename, 'w') as f:
            json.dump(score_AL, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with wandb sweep configuration.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("filename", type=str, nargs='?', default="output", help="Custom name for the output files (optional).")
    
    args = parser.parse_args()
    config_file = "config/" + args.config_file + ".yaml"
    # wandb.login()
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if config["parameters"]["seed"]["value"] == "None":
        config["parameters"]["seed"]["value"] = random.randint(0, 1000000) 
    
    custom_name = args.filename
    pprint.pprint(config)
    # Initialize the sweep
    sweep_id = wandb.sweep(config, project="tests_AL")
    # Run the sweep
    wandb.agent(sweep_id, function=training_loop, count=1)

