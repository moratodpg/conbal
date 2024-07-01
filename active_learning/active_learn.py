import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActiveLearning:
    def __init__(self, num_active_points, budget_total):
        self.num_active_points = num_active_points
        self.budget_total = budget_total
        self.strategy_map = {
            "random": self.get_random_points,
            "random_budget": self.select_random_budget,
            "mutual_info": self.select_mutual_info,
            "mi_adapt_threshold": self.select_mi_adapt_threshold,
            "mi_cost": self.select_mi_cost,
            "entropy": self.select_entropy,
            "var_ratio": self.select_var_ratio,
        }

    ## Wrapper function to get the active points
    def get_points(self, mode, net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1):
        if mode == "random":
            return self.get_random_points(idx_pool, coordinates, cost_factor)
        else:
            strategy = self.strategy_map.get(mode, self.strategy_map["mutual_info"])
            return strategy(net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1)
    
    ## Helper functions
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
    
    ## Strategies are defined below
    def get_random_points(self, idx_pool, coordinates, cost_factor):
        # create a list of random numbers as integers from 0 to len(idx_pool)
        random_idx_pool = np.random.choice(idx_pool, self.num_active_points, replace=False)
        random_idx_pool = random_idx_pool.tolist()
        cost_total = 0
        # initial_coord = torch.tensor([91283, 437631])
        index_initial = random_idx_pool[0]
        for index in random_idx_pool[1:]:
            distance_cost = self.compute_distances(coordinates, index_initial, cost_factor)
            cost_total += distance_cost[index].item()/1000
            index_initial = index
        return random_idx_pool, cost_total
    
    def select_random_budget(self, net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1):
        cost_total = 0        
        # initial_coord = torch.tensor([91283, 437631])
        # distance_cost = self.compute_distances(coordinates[idx_pool], initial_coord, cost_factor)
        budget_points = self.num_active_points - 1
        budget = self.budget_total
        random_idx = np.random.choice(idx_pool, 1, replace=False)
        selected_ind = random_idx.tolist()
        # cost_total += distance_cost[selected_ind[-1]].item()/1000

        for _ in range(self.num_active_points - 1):
            # Masking out points beyond adaptive threshold
            idx_pool = np.setdiff1d(idx_pool, selected_ind)
            distance_cost = self.compute_distances(coordinates[idx_pool], coordinates[selected_ind[-1]], cost_factor)
            budget_threshold = budget / (budget_points)
            distance_cost_mask = distance_cost/1000 < budget_threshold
            possible_indices = idx_pool[distance_cost_mask]
            if len(possible_indices) == 0:
                # Check if all points exceed the total budget
                total_budget_mask = distance_cost/1000 < budget
                possible_indices = idx_pool[total_budget_mask]
                if len(possible_indices) == 0:
                    return selected_ind, cost_total
                random_idx = np.random.choice(possible_indices, 1, replace=False)
            else:
                random_idx = np.random.choice(possible_indices, 1, replace=False)
            
            point_cost = self.compute_distances(coordinates[random_idx], coordinates[selected_ind[-1]], cost_factor).unsqueeze(0)[0]
            cost_total += point_cost.item()/1000 # Not correct
            budget -= point_cost.item()/1000 # Not correct
            budget_points -= 1
            selected_ind.append(random_idx.item())
        # Last point (return cost)
        # cost_total += self.compute_distances(coordinates[idx_pool[selected_ind[-1]]].unsqueeze(0), initial_coord, cost_factor).item()/1000
        return selected_ind, cost_total
    
    def select_mutual_info(self, net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1):
        cost_total = 0
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum
        # initial_coord = torch.tensor([91283, 437631])
        # distance_cost = self.compute_distances(coordinates[idx_pool], initial_coord, cost_factor)
        _, mi_indices = mutual_info.topk(1)
        selected_ind = mi_indices.tolist()
        # cost_total += distance_cost[selected_ind[-1]].item()/1000

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

        # Last point (return cost)
        # cost_total += self.compute_distances(coordinates[idx_pool[selected_ind[-1]]].unsqueeze(0), initial_coord, cost_factor).item()/1000

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total
    
    def select_mi_adapt_threshold(self, net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1):
        cost_total = 0
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum
        # initial_coord = torch.tensor([91283, 437631])
        # distance_cost = self.compute_distances(coordinates[idx_pool], initial_coord, cost_factor)

        budget_points = self.num_active_points - 1
        budget = self.budget_total # To be amended
        
        _, mi_indices = mutual_info.topk(1)
        selected_ind = mi_indices.tolist()
        # cost_total += distance_cost[selected_ind[-1]].item()/1000

        # Evaluate the first point and store it
        selected_predicts = predicts[selected_ind[-1]].unsqueeze(0)
        stored_predicts = selected_predicts.clone()

        for _ in range(self.num_active_points - 1):
            # Compute mutual information
            joint_cond_entropy = entropy_sum + entropy_sum[selected_ind].sum()
            # [samples, num_classes^n-1, num_classes] = [1, num_classes, num_forwards] x [samples, num_forwards, num_classes]
            joint_per_sample = torch.einsum('ijk , bkc -> bjc' , selected_predicts.transpose(1,2) , predicts)
            joint_per_sample = joint_per_sample.reshape(-1, joint_per_sample.shape[1]*joint_per_sample.shape[2])/num_forwards
            eps = 1e-9
            joint_entropy_clamped = torch.clamp(joint_per_sample, min=eps)
            joint_predictive_entropy = -torch.sum(joint_entropy_clamped * torch.log2(joint_entropy_clamped), dim=1)
            joint_mutual_info = joint_predictive_entropy - joint_cond_entropy

            # Masking out points beyond adaptive threshold
            distance_cost = self.compute_distances(coordinates[idx_pool], coordinates[idx_pool[selected_ind[-1]]], cost_factor)
            budget_threshold = budget / (budget_points)
            distance_cost_mask = distance_cost/1000 > budget_threshold

            joint_mutual_info[distance_cost_mask] = 0
            joint_mutual_info[selected_ind] = 0

            # Masking out points beyond total budget; if all exceed it, return collected points
            if joint_mutual_info.sum() == 0:
                joint_mutual_info = joint_predictive_entropy - joint_cond_entropy
                joint_mutual_info[selected_ind] = 0
                distance_cost_mask = distance_cost/1000 > budget
                joint_mutual_info[distance_cost_mask] = 0

                if joint_mutual_info.sum() == 0:
                    selected_idx_pool = [idx_pool[i] for i in selected_ind]
                    return selected_idx_pool, cost_total

            # Select the next batch active point
            _, mi_indices = joint_mutual_info.topk(1)
            selected_ind.append(mi_indices.item())
            cost_total += distance_cost[selected_ind[-1]].item()/1000
            
            budget -= distance_cost[selected_ind[-1]].item()/1000
            budget_points -= 1

            # Store the selected point
            selected_predicts = predicts[selected_ind[-1]]
            selected_predicts = torch.einsum('ik,il->ikl', selected_predicts, stored_predicts.squeeze(0)).reshape(num_forwards, -1).unsqueeze(0)
            stored_predicts = selected_predicts.clone()

        # Last point (return cost)
        # cost_total += self.compute_distances(coordinates[idx_pool[selected_ind[-1]]].unsqueeze(0), initial_coord, cost_factor).item()/1000

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total

    def select_mi_cost(self, net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1):
        cost_total = 0
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum

        # initial_coord = torch.tensor([91283, 437631])
        # distance_cost = self.compute_distances(coordinates[idx_pool], initial_coord, cost_factor)

        # mutual_info_norm = (mutual_info - torch.mean(mutual_info))/torch.std(mutual_info)
        # distance_norm = (distance_cost - torch.mean(distance_cost))/torch.std(distance_cost)
        # mutual_info_norm = (mutual_info - torch.min(mutual_info))/(torch.max(mutual_info) - torch.min(mutual_info))
        # distance_norm = (distance_cost - torch.min(distance_cost))/(torch.max(distance_cost) - torch.min(distance_cost))
        # mutual_info_cost = 4*mutual_info_norm - distance_norm
        
        _, mi_indices = mutual_info.topk(1)
        selected_ind = mi_indices.tolist()
        # cost_total += distance_cost[selected_ind[-1]].item()/1000

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
            
            # joint_mutual_norm = (joint_mutual_info - torch.mean(joint_mutual_info))/torch.std(joint_mutual_info)
            # distance_norm = (distance_cost - torch.mean(distance_cost))/torch.std(distance_cost)
            joint_mutual_norm = (joint_mutual_info - torch.min(joint_mutual_info))/(torch.max(joint_mutual_info) - torch.min(joint_mutual_info))
            distance_norm = (distance_cost - torch.min(distance_cost))/(torch.max(distance_cost) - torch.min(distance_cost))
            joint_mutual_info = 4*joint_mutual_norm - distance_norm
            # joint_mutual_info = joint_mutual_info / distance_cost
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

        # Last point (return cost)
        # cost_total += self.compute_distances(coordinates[idx_pool[selected_ind[-1]]].unsqueeze(0), initial_coord, cost_factor).item()/1000
        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total
    
    def select_entropy(self, net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1):
        cost_total = 0
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        entropy = self.predictive_entropy(predicts)
        
        _, mi_indices = entropy.topk(1)
        selected_ind = mi_indices.tolist()
        # cost_total += distance_cost[selected_ind[-1]].item()/1000

        # Evaluate the first point and store it
        selected_predicts = predicts[selected_ind[-1]].unsqueeze(0)
        stored_predicts = selected_predicts.clone()

        for _ in range(self.num_active_points - 1):
            # [samples, num_classes^n-1, num_classes] = [1, num_classes, num_forwards] x [samples, num_forwards, num_classes]
            joint_per_sample = torch.einsum('ijk , bkc -> bjc' , selected_predicts.transpose(1,2) , predicts)
            joint_per_sample = joint_per_sample.reshape(-1, joint_per_sample.shape[1]*joint_per_sample.shape[2])/num_forwards
            eps = 1e-9
            joint_entropy_clamped = torch.clamp(joint_per_sample, min=eps)
            joint_predictive_entropy = -torch.sum(joint_entropy_clamped * torch.log2(joint_entropy_clamped), dim=1)
            distance_cost = self.compute_distances(coordinates[idx_pool], coordinates[idx_pool[selected_ind[-1]]], cost_factor)
            # Mask already selected indices
            joint_predictive_entropy[selected_ind] = 0
            # Select the next batch active point
            _, mi_indices = joint_predictive_entropy.topk(1)
            selected_ind.append(mi_indices.item())
            cost_total += distance_cost[selected_ind[-1]].item()/1000

            # Store the selected point
            selected_predicts = predicts[selected_ind[-1]]
            selected_predicts = torch.einsum('ik,il->ikl', selected_predicts, stored_predicts.squeeze(0)).reshape(num_forwards, -1).unsqueeze(0)
            stored_predicts = selected_predicts.clone()

        # Last point (return cost)
        # cost_total += self.compute_distances(coordinates[idx_pool[selected_ind[-1]]].unsqueeze(0), initial_coord, cost_factor).item()/1000
        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total
    
    # It needs some updates
    def select_var_ratio(self, net_current, num_forwards, buildings_dataset, idx_pool, coordinates, cost_factor=1):
        cost_total = 0
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        # initial_coord = torch.tensor([91283, 437631])
        # distance_cost = self.compute_distances(coordinates[idx_pool], initial_coord, cost_factor)

        predicted_classes = torch.argmax(predicts, dim=2)
        mode_vals, _ = torch.mode(predicted_classes, dim=1)
        n_c = torch.stack([torch.sum(predicted_classes[i] == mode_vals[i]) for i in range(predicted_classes.shape[0])])
        mutual_info_cost = 1 - n_c / num_forwards
        
        _, mi_indices = mutual_info_cost.topk(1)
        selected_ind = mi_indices.tolist()
        # cost_total += distance_cost[selected_ind[-1]].item()/1000
        # Evaluate the first point and store it
        selected_predicts = predicts[selected_ind[-1]].unsqueeze(0)
        stored_predicts = selected_predicts.clone()

        for _ in range(self.num_active_points - 1):
            # Compute the conditional entropy 
            # print(joint_cond_entropy)
            # [samples, num_classes^n-1, num_classes] = [1, num_classes, num_forwards] x [samples, num_forwards, num_classes]
            joint_per_sample = torch.einsum('ijk , bkc -> bjc' , selected_predicts.transpose(1,2) , predicts)
            joint_per_sample = joint_per_sample.reshape(-1, joint_per_sample.shape[1]*joint_per_sample.shape[2])/num_forwards

            distance_cost = self.compute_distances(coordinates[idx_pool], coordinates[idx_pool[selected_ind[-1]]], cost_factor)
            
            predicted_classes = torch.argmax(predicts, dim=2)
            mode_vals, _ = torch.mode(predicted_classes, dim=1)
            n_c = torch.stack([torch.sum(predicted_classes[i] == mode_vals[i]) for i in range(predicted_classes.shape[0])])
            joint_mutual_info = 1 - n_c / num_forwards
                
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
        # Last point (return cost)
        # cost_total += self.compute_distances(coordinates[idx_pool[selected_ind[-1]]].unsqueeze(0), initial_coord, cost_factor).item()/1000
        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total