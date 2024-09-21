import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
    
class MLP_dropout(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, dropout_prob):
        super(MLP_dropout, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth

        layers = [nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout_prob)]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# Define a class for training and testing the classifier
class Trainer_Ensemble:
    def __init__(self, model, num_classes, train_loader, test_loader, criterion, optimizer, num_epochs, patience=30):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_classes = num_classes
        self.score = 0
        self.best_val_score = float('-inf')
        self.epochs_without_improvement = 0
        self.best_model = None * len(self.models)

    def train(self):
        for epoch in range(self.num_epochs):
            [net.train() for net in self.model]
            running_loss = 0.0
            for inputs, labels, _ in self.train_loader:
                for i, net in enumerate(self.model):
                    self.optimizer[i].zero_grad()
                    outputs = net(inputs)
                    loss = self.criterion(outputs, labels)
                    # print(loss.item())
                    loss.backward()
                    self.optimizer[i].step()
                    running_loss += loss.item()

            # Evaluate the model on the test set to calculate validation loss
            val_score = self.evaluate()

            # Early stopping check
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.epochs_without_improvement = 0
                self.best_model = [copy.deepcopy(net.state_dict()) for net in self.model]
                # print(f"Validation score is now: {val_score:.4f}")
            else:
                self.epochs_without_improvement += 1
                #print(f"Validation loss did not decrease, count: {self.epochs_without_improvement}")
                if self.epochs_without_improvement >= self.patience:
                    print("Early stopping triggered", ", Epoch: " ,epoch + 1)
                    self.test()
                    break

            if epoch == (self.num_epochs - 1):
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {running_loss / len(self.train_loader)}")
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
                ensemble_outputs = [net(inputs) for net in self.model]
                avg_outputs = sum(ensemble_outputs) / len(self.model)
                _, predicted_labels = torch.max(avg_outputs, 1) 
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
        scores = self.compute_scores()
        return scores["accuracy"]

    def test(self):
        for i, net in enumerate(self.model):
            net.load_state_dict(self.best_models_state[i])
        [net.eval() for net in self.model] 
        scores = self.compute_scores()
        self.score = scores["accuracy"]
        # Print the computed metrics
        print(f"Accuracy on the test set: {scores['accuracy']}")
        print(f"Average Precision on the test set: {scores['avg_precision']}")
        print(f"Average Recall on the test set: {scores['avg_recall']}")
        print(f"Average F1 Score on the test set: {scores['avg_f1_score']}")