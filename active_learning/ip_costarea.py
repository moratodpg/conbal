import torch
import torch.nn.functional as F
import pulp

import torch.nn as nn
import numpy as np

from active_learning.active_learning import ActiveLearning

class Ip_Costarea_MI(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
    
    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool):
        cost_total = 0
        budget = self.budget_total
        areacost = self.cost_area[idx_pool]

        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum

        if sum(areacost < budget) < 1:
            selected_idx_pool = []
            print("No budget available for selection")
            return selected_idx_pool, cost_total

        mi_values_np = mutual_info.numpy()
        areacosts_np = areacost.numpy()

        prob = pulp.LpProblem("Maximize_Mutual_Information", pulp.LpMaximize)

        size_variables = len(mutual_info)

        # Define decision variables
        x = pulp.LpVariable.dicts("x", range(size_variables), cat='Binary')
        # Objective function
        prob += pulp.lpSum(mi_values_np[i] * x[i] for i in range(size_variables))
        # Budget constraint
        prob += pulp.lpSum(areacosts_np[i] * x[i] for i in range(size_variables)) <= budget
        # At most 5 items constraint
        prob += pulp.lpSum(x[i] for i in range(size_variables)) <= self.num_active_points
        # Solve the problem with suppressed output
        solver_status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

        selected_ind = [i for i in range(size_variables) if pulp.value(x[i]) == 1]
        cost_total += sum(areacost[i] for i in selected_ind).item()

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total