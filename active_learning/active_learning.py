import torch
import torch.nn.functional as F

import torch.nn as nn
import numpy as np

class ActiveLearning:
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        self.num_active_points = num_active_points
        self.budget_total = budget_total
        self.coordinates = coordinates
        self.cost_area = cost_area

    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool):
        pass

    def predict(self, model, inputs, forward_passes):
        model.train()
        predictions = []
        with torch.no_grad():
            for _ in range(forward_passes):
                outputs = model(inputs)
                predicted = F.softmax(outputs, dim=1)
                predictions.append(predicted)
        predicts = torch.stack(predictions, dim=1)
        return predicts
    
    def predictive_entropy(self, predictions):
        avg_predicts = torch.mean(predictions, dim=1)
        eps = 1e-9
        avg_probs_clamped = torch.clamp(avg_predicts, min=eps)
        predictive_entropy = -torch.sum(avg_probs_clamped * torch.log2(avg_probs_clamped), dim=1)
        return predictive_entropy
    
    def expected_conditional_entropy(self, predictions):
        eps = 1e-9
        prob_clamped = torch.clamp(predictions, min=eps)
        entropy_i = -torch.sum(prob_clamped * torch.log2(prob_clamped), dim=2)
        entropy_sum = torch.mean(entropy_i, dim=1)
        return entropy_sum
    
    def mutual_information(self, predictions):
        predictive_entropy = self.predictive_entropy(predictions)
        expected_conditional_entropy = self.expected_conditional_entropy(predictions)
        mutual_info = predictive_entropy - expected_conditional_entropy
        return mutual_info
    
    def compute_distances(self, buildings_tensor, ref_coord, cost_factor):
        # Compute the differences in coordinates
        differences = buildings_tensor - ref_coord
        # Compute the squared differences and sum them along the last dimension
        squared_differences = differences ** 2
        squared_distances = squared_differences.sum(dim=1)
        # Compute the square root of the squared distances to get the Euclidean distances
        distances = torch.sqrt(squared_distances)*cost_factor
        return distances