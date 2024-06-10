import torch.nn as nn
import torch.nn.functional as F
import torch
import copy


# Define a simple neural network for classification
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        # self.fc0 = nn.Linear(input_size, num_classes)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)
        self.fc3 = nn.Linear(2, num_classes)

        
    def forward(self, x):
        # x = self.fc0(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def binary_output(self, x):
        # x = self.fc0(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
    
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
    


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=None)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=None)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        

# Define a class for training and testing the classifier
class Trainer:
    def __init__(self, model, num_classes, train_loader, test_loader, criterion, optimizer, num_epochs, patience=30):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_classes = num_classes
        self.f1_score = 0
        self.best_val_score = float('-inf')
        self.epochs_without_improvement = 0
        self.best_model = None

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            for inputs, labels, _ in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                print(loss.item())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(self.train_loader)

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
        # Print the computed metrics
        print(f"Accuracy on the test set: {scores['accuracy']}")
        print(f"Average Precision on the test set: {scores['avg_precision']}")
        print(f"Average Recall on the test set: {scores['avg_recall']}")
        print(f"Average F1 Score on the test set: {scores['avg_f1_score']}")

# Define a class for training and testing the classifier
class MultiClassTrainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, num_epochs, n_classes):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.n_classes = n_classes

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            for inputs, labels, _ in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Print average loss for the epoch
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {running_loss / len(self.train_loader)}")

            # Test the model after each epoch
            self.test()

    def test(self):
        self.model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        TP, FP, FN = [0] * self.n_classes, [0] * self.n_classes, [0] * self.n_classes
        with torch.no_grad():
            for inputs, labels, _ in self.test_loader:
                outputs = self.model(inputs)
                _, predicted_labels = torch.max(outputs, 1) 
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()
                
                for i in range(self.n_classes):
                    TP[i] += ((predicted_labels == i) & (labels == i)).sum().item()
                    FP[i] += ((predicted_labels == i) & (labels != i)).sum().item()
                    FN[i] += ((predicted_labels != i) & (labels == i)).sum().item()
        
        # Compute precision, recall, and F1 score for each class
        precision = [TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0 for i in range(self.n_classes)]
        recall = [TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0 for i in range(self.n_classes)]
        f1_score = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0 for i in range(self.n_classes)]

        # To compute the average across classes
        avg_precision = sum(precision) / self.n_classes
        avg_recall = sum(recall) / self.n_classes
        avg_f1_score = sum(f1_score) / self.n_classes

        # Overall accuracy
        accuracy = correct / total if total > 0 else 0

        # Print accuracy on the test set
        # accuracy = correct / total
        # print(f"Accuracy on the test set: {accuracy}")

        # Print the computed metrics
        print(f"Accuracy on the test set: {accuracy}")
        print(f"Average Precision on the test set: {avg_precision}")
        print(f"Average Recall on the test set: {avg_recall}")
        print(f"Average F1 Score on the test set: {avg_f1_score}")