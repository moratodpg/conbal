import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActiveLearning:
    def __init__(self, num_active_points):
        self.num_active_points = num_active_points

    def get_points(self, mode, independent_mode, net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1):
        if mode == "random":
            return self.get_random_points(idx_pool, coordinates, cost_factor)
        else:
            learning_option = [mode, independent_mode]
            return self.get_active_points(learning_option, net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1)

    def get_random_points(self, idx_pool, coordinates, cost_factor):
        # create a list of random numbers as integers from 0 to len(idx_pool)
        random_idx_pool = np.random.choice(idx_pool, self.num_active_points, replace=False)
        random_idx_pool = random_idx_pool.tolist()
        cost_total = 0
        initial_coord = torch.tensor([91283, 437631])
        for index in random_idx_pool:
            distance_cost = self.compute_distances(coordinates, initial_coord, cost_factor)
            initial_coord = coordinates[index]
            cost_total += distance_cost[index].item()/1000
        return random_idx_pool, cost_total
    
    def get_active_points(self, learning_option, net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1):
        cost_total = 0

        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum

        initial_coord = torch.tensor([91283, 437631])
        distance_cost = self.compute_distances(coordinates[idx_pool], initial_coord, cost_factor)

        if learning_option[0] == "mutual_info_cost":
            # mutual_info_norm = (mutual_info - torch.mean(mutual_info))/torch.std(mutual_info)
            # distance_norm = (distance_cost - torch.mean(distance_cost))/torch.std(distance_cost)
            mutual_info_norm = (mutual_info - torch.min(mutual_info))/(torch.max(mutual_info) - torch.min(mutual_info))
            distance_norm = (distance_cost - torch.min(distance_cost))/(torch.max(distance_cost) - torch.min(distance_cost))
            mutual_info_cost = 4*mutual_info_norm - distance_norm
        elif learning_option[0] == "mutual_info":
            mutual_info_cost = mutual_info
        elif learning_option[0] == "entropy":
            mutual_info_cost = entropy
        elif learning_option[0] == "var_ratio":
            predicted_classes = torch.argmax(predicts, dim=2)
            mode_vals, _ = torch.mode(predicted_classes, dim=1)
            n_c = torch.stack([torch.sum(predicted_classes[i] == mode_vals[i]) for i in range(predicted_classes.shape[0])])
            mutual_info_cost = 1 - n_c / num_forwards
        elif learning_option[0] == "cost":
            mutual_info_cost = -distance_cost

        _, mi_indices = mutual_info_cost.topk(1)
        selected_ind = mi_indices.tolist()

        if learning_option[1]:
            _, mi_indices = mutual_info_cost.topk(self.num_active_points)
            selected_ind = mi_indices.tolist()
            for index in selected_ind:
                cost_total += distance_cost[index].item()/1000
            # From the selected indices, get the indices from the pool
            selected_idx_pool = [idx_pool[i] for i in selected_ind]
            return selected_idx_pool, cost_total

        cost_total += distance_cost[selected_ind[-1]].item()/1000

        # Evaluate the first point and store it
        selected_predicts = predicts[selected_ind[-1]].unsqueeze(0)
        stored_predicts = selected_predicts.clone()

        for _ in range(self.num_active_points - 1):

            # Compute the conditional entropy 
            joint_cond_entropy = entropy_sum + entropy_sum[selected_ind].sum()
            # print(joint_cond_entropy)

            # [samples, num_classes^n-1, num_classes] = [1, num_classes, num_forwards] x [samples, num_forwards, num_classes]
            joint_per_sample = torch.einsum('ijk , bkc -> bjc' , selected_predicts.transpose(1,2) , predicts)
            joint_per_sample = joint_per_sample.reshape(-1, joint_per_sample.shape[1]*joint_per_sample.shape[2])/num_forwards

            eps = 1e-9
            joint_entropy_clamped = torch.clamp(joint_per_sample, min=eps)
            joint_predictive_entropy = -torch.sum(joint_entropy_clamped * torch.log2(joint_entropy_clamped), dim=1)

            joint_mutual_info = joint_predictive_entropy - joint_cond_entropy

            distance_cost = self.compute_distances(coordinates[idx_pool], coordinates[idx_pool[selected_ind[-1]]], cost_factor)
            if learning_option[0] == "mutual_info_cost":
                # joint_mutual_norm = (joint_mutual_info - torch.mean(joint_mutual_info))/torch.std(joint_mutual_info)
                # distance_norm = (distance_cost - torch.mean(distance_cost))/torch.std(distance_cost)
                joint_mutual_norm = (joint_mutual_info - torch.min(joint_mutual_info))/(torch.max(joint_mutual_info) - torch.min(joint_mutual_info))
                distance_norm = (distance_cost - torch.min(distance_cost))/(torch.max(distance_cost) - torch.min(distance_cost))
                joint_mutual_info = 4*joint_mutual_norm - distance_norm
                # joint_mutual_info = joint_mutual_info / distance_cost
            elif learning_option[0] == "entropy":
                joint_mutual_info = joint_predictive_entropy
            elif learning_option[0] == "var_ratio":
                predicted_classes = torch.argmax(predicts, dim=2)
                mode_vals, _ = torch.mode(predicted_classes, dim=1)
                n_c = torch.stack([torch.sum(predicted_classes[i] == mode_vals[i]) for i in range(predicted_classes.shape[0])])
                joint_mutual_info = 1 - n_c / num_forwards
            elif learning_option[0] == "cost":
                joint_mutual_info = -distance_cost
                
            # Mask already selected indices
            joint_mutual_info[selected_ind] = 0

            # Select the next batch active point
            _, mi_indices = joint_mutual_info.topk(1)
            selected_ind.append(mi_indices.item())
            cost_total += distance_cost[selected_ind[-1]].item()/1000

            # Store the selected point
            selected_predicts = predicts[selected_ind[-1]]
            selected_predicts = torch.einsum('ik,il->ikl', selected_predicts, stored_predicts.squeeze(0)).reshape(num_forwards, -1).unsqueeze(0)
            stored_predicts = selected_predicts.clone()

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]

        return selected_idx_pool, cost_total
    
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