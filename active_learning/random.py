import torch
import torch.nn.functional as F

import torch.nn as nn
import numpy as np

from active_learning.active_learning import ActiveLearning

class RandomBudget(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool, idx_train):
        cost_total = 0
        cost_factor = 1
        budget = self.budget_total 

        random_idx = np.random.choice(idx_pool, 1, replace=False)
        selected_ind = random_idx.tolist()

        for _ in range(self.num_active_points - 1):
            idx_pool = np.setdiff1d(idx_pool, selected_ind)
            distance_cost = self.compute_distances(self.coordinates[idx_pool], self.coordinates[selected_ind[-1]], cost_factor)
            distance_cost_mask = distance_cost/1000 < budget
            possible_indices = idx_pool[distance_cost_mask]

            if len(possible_indices) == 0:
                return selected_ind, cost_total
            
            random_idx = np.random.choice(possible_indices, 1, replace=False)
            point_cost = self.compute_distances(self.coordinates[random_idx], self.coordinates[selected_ind[-1]], cost_factor).unsqueeze(0)[0]
            cost_total += point_cost.item()/1000 # Not correct
            budget -= point_cost.item()/1000 # Not correct
            selected_ind.append(random_idx.item())

        return selected_ind, cost_total
    
class RandomBudgetReturn(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool, idx_train):
        cost_total = 0
        cost_factor = 1
        budget = self.budget_total 

        random_idx = np.random.choice(idx_pool, 1, replace=False)
        selected_ind = random_idx.tolist()
        coord_start = self.coordinates[selected_ind[-1]]

        for _ in range(self.num_active_points - 1):
            idx_pool = np.setdiff1d(idx_pool, selected_ind)
            distance_cost = self.compute_distances(self.coordinates[idx_pool], self.coordinates[selected_ind[-1]], cost_factor)
            cost_return = self.compute_distances(self.coordinates[idx_pool], coord_start, cost_factor)
            cost_return += distance_cost
            
            distance_cost_mask = cost_return/1000 < budget
            possible_indices = idx_pool[distance_cost_mask]

            if len(possible_indices) == 0:
                return_cost = self.compute_distances(self.coordinates[selected_ind[-1]].unsqueeze(0), coord_start, cost_factor)[0]
                cost_total += return_cost.item()/1000
                return selected_ind, cost_total
            
            random_idx = np.random.choice(possible_indices, 1, replace=False)
            point_cost = self.compute_distances(self.coordinates[random_idx], self.coordinates[selected_ind[-1]], cost_factor).unsqueeze(0)[0]
            cost_total += point_cost.item()/1000 # Not correct
            budget -= point_cost.item()/1000 # Not correct
            selected_ind.append(random_idx.item())
        
        # Last point return cost
        return_cost = self.compute_distances(self.coordinates[selected_ind[-1]].unsqueeze(0), coord_start, cost_factor)[0]
        cost_total += return_cost.item()/1000

        return selected_ind, cost_total
    
class RandomAreas(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool, idx_train):
        cost_total = 0
        budget = self.budget_total 
        selected_ind = []

        for _ in range(self.num_active_points):
            idx_pool = np.setdiff1d(idx_pool, selected_ind)

            area_cost = self.cost_area[idx_pool]
            area_cost_mask = area_cost < budget
            possible_indices = idx_pool[area_cost_mask]

            if len(possible_indices) == 0:
                return selected_ind, cost_total
            
            random_idx = np.random.choice(possible_indices, 1, replace=False)
            point_cost = self.cost_area[random_idx]
            cost_total += point_cost.item()
            budget -= point_cost.item()
            selected_ind.append(random_idx.item())

        return selected_ind, cost_total