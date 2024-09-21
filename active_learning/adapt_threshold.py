import torch
import torch.nn.functional as F

import torch.nn as nn
import numpy as np

from active_learning.active_learning import ActiveLearning

class MIAdaptThreshold(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool):
        cost_total = 0
        cost_factor = 1
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum

        budget_points = self.num_active_points - 1
        budget = self.budget_total 
        
        _, mi_indices = mutual_info.topk(1)
        selected_ind = mi_indices.tolist()

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
            distance_cost = self.compute_distances(self.coordinates[idx_pool], self.coordinates[idx_pool[selected_ind[-1]]], cost_factor)
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

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total
    
    
class MIAdaptThresholdReturn(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool):
        cost_total = 0
        cost_factor = 1
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum

        budget_points = self.num_active_points
        budget = self.budget_total 
        
        _, mi_indices = mutual_info.topk(1)
        selected_ind = mi_indices.tolist()
        coord_start = self.coordinates[idx_pool[selected_ind[-1]]]

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
            distance_cost = self.compute_distances(self.coordinates[idx_pool], self.coordinates[idx_pool[selected_ind[-1]]], cost_factor)
            budget_threshold = budget / (budget_points)
            distance_cost_mask = distance_cost/1000 > budget_threshold

            # Masking out points considering return to initial point
            cost_return = self.compute_distances(self.coordinates[idx_pool], coord_start, cost_factor)
            cost_return += distance_cost
            cost_return_mask = cost_return/1000 > budget

            joint_mutual_info[distance_cost_mask] = 0
            joint_mutual_info[cost_return_mask] = 0
            joint_mutual_info[selected_ind] = 0

            # Masking out points beyond total budget; if all exceed it, return collected points
            if joint_mutual_info.sum() == 0:
                joint_mutual_info = joint_predictive_entropy - joint_cond_entropy
                joint_mutual_info[selected_ind] = 0
                joint_mutual_info[cost_return_mask] = 0

                if joint_mutual_info.sum() == 0:
                    selected_idx_pool = [idx_pool[i] for i in selected_ind]
                    return_cost = self.compute_distances(self.coordinates[idx_pool][selected_ind[-1]].unsqueeze(0), coord_start, cost_factor)[0]
                    cost_total += return_cost.item()/1000
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
        return_cost = self.compute_distances(self.coordinates[idx_pool][selected_ind[-1]].unsqueeze(0), coord_start, cost_factor)[0]
        cost_total += return_cost.item()/1000

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total


class MIAdaptThresholdArea(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool):
        cost_total = 0
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum

        budget_points = self.num_active_points 
        budget = self.budget_total 
        budget_threshold = budget / (budget_points)

        area_cost = self.cost_area[idx_pool]
        area_cost_mask = area_cost > budget_threshold
        mutual_info[area_cost_mask] = 0

        if mutual_info.sum() == 0:
            mutual_info = entropy - entropy_sum
            area_cost_mask = area_cost > budget
            mutual_info[area_cost_mask] = 0
        
        _, mi_indices = mutual_info.topk(1)
        selected_ind = mi_indices.tolist()
        cost_total += area_cost[selected_ind[-1]].item()
        budget -= area_cost[selected_ind[-1]].item()
        budget_points -= 1

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
            budget_threshold = budget / (budget_points)
            area_cost_mask = area_cost > budget_threshold

            joint_mutual_info[area_cost_mask] = 0
            joint_mutual_info[selected_ind] = 0

            # Masking out points beyond total budget; if all exceed it, return collected points
            if joint_mutual_info.sum() == 0:
                joint_mutual_info = joint_predictive_entropy - joint_cond_entropy
                joint_mutual_info[selected_ind] = 0
                area_cost_mask = area_cost > budget
                joint_mutual_info[area_cost_mask] = 0

                if joint_mutual_info.sum() == 0:
                    selected_idx_pool = [idx_pool[i] for i in selected_ind]
                    return selected_idx_pool, cost_total

            # Select the next batch active point
            _, mi_indices = joint_mutual_info.topk(1)
            selected_ind.append(mi_indices.item())
            cost_total += area_cost[selected_ind[-1]].item()
            
            budget -= area_cost[selected_ind[-1]].item()
            budget_points -= 1

            # Store the selected point
            selected_predicts = predicts[selected_ind[-1]]
            selected_predicts = torch.einsum('ik,il->ikl', selected_predicts, stored_predicts.squeeze(0)).reshape(num_forwards, -1).unsqueeze(0)
            stored_predicts = selected_predicts.clone()

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total