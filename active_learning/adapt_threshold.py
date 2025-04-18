import torch
import torch.nn.functional as F

import torch.nn as nn
import numpy as np

from active_learning.active_learning import ActiveLearning

### MI (dropout) ###
class MIAdaptThreshold(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool, idx_train):
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
        
    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool,idx_train):
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
        
    def get_points(self, net_current, num_forwards, buildings_dataset, idx_pool, idx_train):
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
            if mutual_info.sum() == 0:
                selected_idx_pool = []
                return selected_idx_pool, cost_total
        
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
    
### Badge ###
class Adapt_Threshold_Badge(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, _, buildings_dataset, idx_pool, idx_train):
        cost_total = 0
        cost_factor = 1

        net_current.eval()
        gradients_matrix = self.per_sample_grad_last_layer(net_current, buildings_dataset.input_tensor[idx_pool])
        net_current.train()

        N, D = gradients_matrix.shape

        # Step 1: choose the first center randomly
        first_idx = torch.randint(high=N, size=(1,)).item()
        # Step 2: compute the distance from the first center
        center_indices = [first_idx]
        centers = [gradients_matrix[first_idx].view(1, D)]

        budget_points = self.num_active_points - 1
        budget = self.budget_total 

        for _ in range(self.num_active_points - 1):
            
            existing_centers = torch.cat(centers, dim=0)  # shape: [c, D], c < k
            # compute L2 distance from each point to each existing center
            # shape: [N, c]
            dists = torch.cdist(gradients_matrix, existing_centers, p=2)
            # for each point, find distance to its closest center
            min_dist, _ = dists.min(dim=1)  # shape: [N]
            # squared distances
            min_dist_sq = min_dist**2
            # pick a new center index with probability proportional to dist^2
            probs = min_dist_sq / torch.sum(min_dist_sq)

            # Masking out points beyond adaptive threshold
            distance_cost = self.compute_distances(self.coordinates[idx_pool], self.coordinates[idx_pool[center_indices[-1]]], cost_factor)
            budget_threshold = budget / (budget_points)
            distance_cost_mask = distance_cost/1000 > budget_threshold

            probs[distance_cost_mask] = 0
            probs[center_indices] = 0

            if probs.sum() == 0:
                probs = min_dist_sq / torch.sum(min_dist_sq)
                probs[center_indices] = 0
                distance_cost_mask = distance_cost/1000 > budget
                probs[distance_cost_mask] = 0

                if probs.sum() == 0:
                    selected_idx_pool = [idx_pool[i] for i in center_indices]
                    return selected_idx_pool, cost_total

            # Select the next batch active point
            new_center_idx = torch.multinomial(probs, 1).item()
            center_indices.append(new_center_idx)
            centers.append(gradients_matrix[new_center_idx].view(1, D))

            cost_total += distance_cost[center_indices[-1]].item()/1000
            budget -= distance_cost[center_indices[-1]].item()/1000
            budget_points -= 1

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in center_indices]
        return selected_idx_pool, cost_total
    
    def per_sample_grad_last_layer(self, model, X):
        """
        Computes per-sample gradient wrt. the LAST LINEAR layer for
        a dropout MLP that ends with a linear(width -> output_dim).

        Using 'predicted labels' (argmax) as the "true" label
        (the 'pseudo-label' scenario).
        """

        # --- 1) Split the final layer from the rest of the MLP ---
        final_linear = model.block[-1]       # e.g. nn.Linear(width, output_dim)
        # all layers except the last
        block_before_final = model.block[:-1]

        # --- 2) Forward pass ---
        # We get the hidden representation from everything except the final linear.
        hidden = block_before_final(X)       # shape: [N, width]
        logits = final_linear(hidden)        # shape: [N, output_dim]

        # Predicted labels => make one-hot
        predicted_labels = logits.argmax(dim=1)                # shape: [N]
        y_onehot = F.one_hot(predicted_labels, logits.size(1)) # shape: [N, output_dim]
        y_onehot = y_onehot.float()

        # Softmax probabilities
        probs = torch.softmax(logits, dim=1) # shape: [N, output_dim]
        # Delta = (p_i - y_i)
        delta = probs - y_onehot            # shape: [N, output_dim]

        # --- 3) Compute final-layer gradients (vectorized) ---
        # final_linear has: W shape = [output_dim, width], b shape = [output_dim]
        # For each sample i:
        #   grad_w[i] = outer(delta[i], hidden[i])  => shape [output_dim, width]
        #   grad_b[i] = delta[i]                    => shape [output_dim]

        # Vectorize the outer products
        grad_w = delta.unsqueeze(2) * hidden.unsqueeze(1)   # shape: [N, output_dim, width]
        grad_w_flat = grad_w.view(X.size(0), -1)            # shape: [N, output_dim*width]

        grad_b = delta                                     # shape: [N, output_dim]

        # Concatenate [grad_w, grad_b] along dim=1
        grad_last_layer = torch.cat([grad_w_flat, grad_b], dim=1)
        # shape => [N, (output_dim * width + output_dim)]

        return grad_last_layer.detach()
    
class Adapt_Threshold_Badge_Area(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, _, buildings_dataset, idx_pool, idx_train):
        cost_total = 0
        cost_factor = 1

        net_current.eval()
        gradients_matrix = self.per_sample_grad_last_layer(net_current, buildings_dataset.input_tensor[idx_pool])
        net_current.train()

        N, D = gradients_matrix.shape

        budget_points = self.num_active_points 
        budget = self.budget_total 
        budget_threshold = budget / (budget_points)

        area_cost = self.cost_area[idx_pool]
        area_cost_mask = area_cost > budget_threshold

        allowed = torch.where(~area_cost_mask)[0] 
        if len(allowed) == 0:
            area_cost_mask = area_cost > budget
            allowed = torch.where(~area_cost_mask)[0] 

            if len(allowed) == 0:
                selected_idx_pool = []
                return selected_idx_pool, cost_total

        # Step 1: choose the first center randomly
        first_idx = allowed[ torch.randint(len(allowed), (1,)).item() ]
        # Step 2: compute the distance from the first center
        center_indices = [first_idx.item()]
        centers = [gradients_matrix[first_idx].view(1, D)]

        cost_total += area_cost[center_indices[-1]].item()
        budget -= area_cost[center_indices[-1]].item()
        budget_points -= 1


        for _ in range(self.num_active_points - 1):
            
            existing_centers = torch.cat(centers, dim=0)  # shape: [c, D], c < k
            # compute L2 distance from each point to each existing center
            # shape: [N, c]
            dists = torch.cdist(gradients_matrix, existing_centers, p=2)
            # for each point, find distance to its closest center
            min_dist, _ = dists.min(dim=1)  # shape: [N]
            # squared distances
            min_dist_sq = min_dist**2
            # pick a new center index with probability proportional to dist^2
            probs = min_dist_sq / torch.sum(min_dist_sq)

            # Masking out points beyond adaptive threshold
            budget_threshold = budget / (budget_points)
            area_cost_mask = area_cost > budget_threshold

            probs[area_cost_mask] = 0
            probs[center_indices] = 0

            if probs.sum() == 0:
                probs = min_dist_sq / torch.sum(min_dist_sq)
                probs[center_indices] = 0
                area_cost_mask = area_cost > budget
                probs[area_cost_mask] = 0

                if probs.sum() == 0:
                    selected_idx_pool = [idx_pool[i] for i in center_indices]
                    return selected_idx_pool, cost_total

            # Select the next batch active point
            new_center_idx = torch.multinomial(probs, 1).item()
            center_indices.append(new_center_idx)
            centers.append(gradients_matrix[new_center_idx].view(1, D))

            cost_total += area_cost[center_indices[-1]].item()
            budget -= area_cost[center_indices[-1]].item()
            budget_points -= 1

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in center_indices]
        return selected_idx_pool, cost_total
    
    def per_sample_grad_last_layer(self, model, X):
        """
        Computes per-sample gradient wrt. the LAST LINEAR layer for
        a dropout MLP that ends with a linear(width -> output_dim).

        Using 'predicted labels' (argmax) as the "true" label
        (the 'pseudo-label' scenario).
        """

        # --- 1) Split the final layer from the rest of the MLP ---
        final_linear = model.block[-1]       # e.g. nn.Linear(width, output_dim)
        # all layers except the last
        block_before_final = model.block[:-1]

        # --- 2) Forward pass ---
        # We get the hidden representation from everything except the final linear.
        hidden = block_before_final(X)       # shape: [N, width]
        logits = final_linear(hidden)        # shape: [N, output_dim]

        # Predicted labels => make one-hot
        predicted_labels = logits.argmax(dim=1)                # shape: [N]
        y_onehot = F.one_hot(predicted_labels, logits.size(1)) # shape: [N, output_dim]
        y_onehot = y_onehot.float()

        # Softmax probabilities
        probs = torch.softmax(logits, dim=1) # shape: [N, output_dim]
        # Delta = (p_i - y_i)
        delta = probs - y_onehot            # shape: [N, output_dim]

        # --- 3) Compute final-layer gradients (vectorized) ---
        # final_linear has: W shape = [output_dim, width], b shape = [output_dim]
        # For each sample i:
        #   grad_w[i] = outer(delta[i], hidden[i])  => shape [output_dim, width]
        #   grad_b[i] = delta[i]                    => shape [output_dim]

        # Vectorize the outer products
        grad_w = delta.unsqueeze(2) * hidden.unsqueeze(1)   # shape: [N, output_dim, width]
        grad_w_flat = grad_w.view(X.size(0), -1)            # shape: [N, output_dim*width]

        grad_b = delta                                     # shape: [N, output_dim]

        # Concatenate [grad_w, grad_b] along dim=1
        grad_last_layer = torch.cat([grad_w_flat, grad_b], dim=1)
        # shape => [N, (output_dim * width + output_dim)]

        return grad_last_layer.detach()
    

### Coreset ###
class Adapt_Threshold_Coreset(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, _, buildings_dataset, idx_pool, idx_train):
        cost_total = 0
        cost_factor = 1

        input_labeled = buildings_dataset.input_tensor[idx_train]
        input_unlabeled = buildings_dataset.input_tensor[idx_pool]
        net_current.eval()
        with torch.no_grad():
            emb_labeled = net_current.block[:-1](input_labeled)
            emb_unlabeled = net_current.block[:-1](input_unlabeled)
        net_current.train()

        dists = torch.cdist(emb_unlabeled, emb_labeled, p=2)
        min_dist, _ = dists.min(dim=1)
        
        _, first_idx = min_dist.topk(1)        
        selected_ind = [first_idx.item()]

        budget = self.budget_total 
        budget_points = self.num_active_points - 1

        for _ in range(self.num_active_points - 1):
            
            # add the selected point to the labeled set
            emb_labeled = torch.cat([emb_labeled, emb_unlabeled[selected_ind[-1]].unsqueeze(0)], dim=0)
            dists = torch.cdist(emb_unlabeled, emb_labeled, p=2)
            min_dist, _ = dists.min(dim=1)

            # Masking out points beyond adaptive threshold
            distance_cost = self.compute_distances(self.coordinates[idx_pool], self.coordinates[idx_pool[selected_ind[-1]]], cost_factor)
            budget_threshold = budget / (budget_points)
            distance_cost_mask = distance_cost/1000 > budget_threshold

            min_dist[distance_cost_mask] = 0
            min_dist[selected_ind] = 0

            if min_dist.sum() == 0:
                min_dist, _ = dists.min(dim=1)
                min_dist[selected_ind] = 0
                distance_cost_mask = distance_cost/1000 > budget
                min_dist[distance_cost_mask] = 0

                if min_dist.sum() == 0:
                    selected_idx_pool = [idx_pool[i] for i in selected_ind]
                    return selected_idx_pool, cost_total

            # Select the next batch active point
            _, sel_idx = min_dist.topk(1)        
            selected_ind.append(sel_idx.item())

            cost_total += distance_cost[selected_ind[-1]].item()/1000
            budget -= distance_cost[selected_ind[-1]].item()/1000
            budget_points -= 1

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total
    
class Adapt_Threshold_Coreset_Area(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, net_current, _, buildings_dataset, idx_pool, idx_train):
        cost_total = 0
        cost_factor = 1

        input_labeled = buildings_dataset.input_tensor[idx_train]
        input_unlabeled = buildings_dataset.input_tensor[idx_pool]
        net_current.eval()
        with torch.no_grad():
            emb_labeled = net_current.block[:-1](input_labeled)
            emb_unlabeled = net_current.block[:-1](input_unlabeled)
        net_current.train()

        dists = torch.cdist(emb_unlabeled, emb_labeled, p=2)
        min_dist, _ = dists.min(dim=1)

        budget_points = self.num_active_points 
        budget = self.budget_total 
        budget_threshold = budget / (budget_points)

        area_cost = self.cost_area[idx_pool]
        area_cost_mask = area_cost > budget_threshold
        min_dist[area_cost_mask] = 0

        if min_dist.sum() == 0:
            min_dist, _ = dists.min(dim=1)
            area_cost_mask = area_cost > budget
            min_dist[area_cost_mask] = 0

            if min_dist.sum() == 0:
                selected_idx_pool = []
                return selected_idx_pool, cost_total

        
        _, first_idx = min_dist.topk(1)        
        selected_ind = [first_idx.item()]

        cost_total += area_cost[selected_ind[-1]].item()
        budget -= area_cost[selected_ind[-1]].item()
        budget_points -= 1

        for _ in range(self.num_active_points - 1):
            
            # add the selected point to the labeled set
            emb_labeled = torch.cat([emb_labeled, emb_unlabeled[selected_ind[-1]].unsqueeze(0)], dim=0)
            dists = torch.cdist(emb_unlabeled, emb_labeled, p=2)
            min_dist, _ = dists.min(dim=1)

            budget_threshold = budget / (budget_points)
            area_cost_mask = area_cost > budget_threshold

            min_dist[area_cost_mask] = 0
            min_dist[selected_ind] = 0

            if min_dist.sum() == 0:
                min_dist, _ = dists.min(dim=1)
                min_dist[selected_ind] = 0
                area_cost_mask = area_cost > budget
                min_dist[area_cost_mask] = 0

                if min_dist.sum() == 0:
                    selected_idx_pool = [idx_pool[i] for i in selected_ind]
                    return selected_idx_pool, cost_total

            # Select the next batch active point
            _, sel_idx = min_dist.topk(1)        
            selected_ind.append(sel_idx.item())

            cost_total += area_cost[selected_ind[-1]].item()
            budget -= area_cost[selected_ind[-1]].item()
            budget_points -= 1

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total
    

### MI (MCMC) ###
class MI_MCMC_Adapt_Threshold(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, trainer_mcmc, buildings_dataset, idx_pool, n_samples=100):
        cost_total = 0
        cost_factor = 1
        predicts = trainer_mcmc.predict_posterior_samples(buildings_dataset.input_tensor[idx_pool])
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum

        budget = self.budget_total 
        budget_points = self.num_active_points - 1
        
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
            joint_per_sample = joint_per_sample.reshape(-1, joint_per_sample.shape[1]*joint_per_sample.shape[2])/n_samples
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
            selected_predicts = torch.einsum('ik,il->ikl', selected_predicts, stored_predicts.squeeze(0)).reshape(n_samples, -1).unsqueeze(0)
            stored_predicts = selected_predicts.clone()

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total
    
class MI_MCMC_Adapt_Threshold_Area(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, trainer_mcmc, buildings_dataset, idx_pool, n_samples=100):
        cost_total = 0
        cost_factor = 1
        predicts = trainer_mcmc.predict_posterior_samples(buildings_dataset.input_tensor[idx_pool])
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
            if mutual_info.sum() == 0:
                selected_idx_pool = []
                return selected_idx_pool, cost_total

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
            joint_per_sample = joint_per_sample.reshape(-1, joint_per_sample.shape[1]*joint_per_sample.shape[2])/n_samples
            eps = 1e-9
            joint_entropy_clamped = torch.clamp(joint_per_sample, min=eps)
            joint_predictive_entropy = -torch.sum(joint_entropy_clamped * torch.log2(joint_entropy_clamped), dim=1)
            joint_mutual_info = joint_predictive_entropy - joint_cond_entropy

            # Masking out points beyond adaptive threshold
            budget_threshold = budget / (budget_points)
            area_cost_mask = area_cost > budget_threshold

            joint_mutual_info[area_cost_mask] = 0
            joint_mutual_info[selected_ind] = 0

            if joint_mutual_info.sum() == 0:
                joint_mutual_info = joint_predictive_entropy - joint_cond_entropy
                joint_mutual_info[selected_ind] = 0
                distance_cost_mask = area_cost > budget
                joint_mutual_info[distance_cost_mask] = 0
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
            selected_predicts = torch.einsum('ik,il->ikl', selected_predicts, stored_predicts.squeeze(0)).reshape(n_samples, -1).unsqueeze(0)
            stored_predicts = selected_predicts.clone()

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total
    
### MI (Ensemble) ###
class MI_ensemble_Adapt_Threshold(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, trainer_ensemble, buildings_dataset, idx_pool, n_ensembles):
        cost_total = 0
        cost_factor = 1
        predicts = trainer_ensemble.predict(buildings_dataset.input_tensor[idx_pool])
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum

        budget = self.budget_total 
        budget_points = self.num_active_points - 1
        
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
            joint_per_sample = joint_per_sample.reshape(-1, joint_per_sample.shape[1]*joint_per_sample.shape[2])/n_ensembles
            eps = 1e-9
            joint_entropy_clamped = torch.clamp(joint_per_sample, min=eps)
            joint_predictive_entropy = -torch.sum(joint_entropy_clamped * torch.log2(joint_entropy_clamped), dim=1)
            joint_mutual_info = joint_predictive_entropy - joint_cond_entropy

            # Masking out points beyond adaptive threshold
            # Masking out points beyond adaptive threshold
            distance_cost = self.compute_distances(self.coordinates[idx_pool], self.coordinates[idx_pool[selected_ind[-1]]], cost_factor)
            budget_threshold = budget / (budget_points)
            distance_cost_mask = distance_cost/1000 > budget_threshold

            joint_mutual_info[distance_cost_mask] = 0
            joint_mutual_info[selected_ind] = 0

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
            selected_predicts = torch.einsum('ik,il->ikl', selected_predicts, stored_predicts.squeeze(0)).reshape(n_ensembles, -1).unsqueeze(0)
            stored_predicts = selected_predicts.clone()

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total
    
class MI_ensemble_Adapt_Threshold_Area(ActiveLearning):
    def __init__(self, num_active_points, budget_total, coordinates, cost_area):
        super().__init__(num_active_points, budget_total, coordinates, cost_area)
        
    def get_points(self, trainer_ensemble, buildings_dataset, idx_pool, n_ensembles):
        cost_total = 0
        cost_factor = 1
        predicts = trainer_ensemble.predict(buildings_dataset.input_tensor[idx_pool])
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
            if mutual_info.sum() == 0:
                selected_idx_pool = []
                return selected_idx_pool, cost_total
        
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
            joint_per_sample = joint_per_sample.reshape(-1, joint_per_sample.shape[1]*joint_per_sample.shape[2])/n_ensembles
            eps = 1e-9
            joint_entropy_clamped = torch.clamp(joint_per_sample, min=eps)
            joint_predictive_entropy = -torch.sum(joint_entropy_clamped * torch.log2(joint_entropy_clamped), dim=1)
            joint_mutual_info = joint_predictive_entropy - joint_cond_entropy

            # Masking out points beyond adaptive threshold
            budget_threshold = budget / (budget_points)
            area_cost_mask = area_cost > budget_threshold

            joint_mutual_info[area_cost_mask] = 0
            joint_mutual_info[selected_ind] = 0

            if joint_mutual_info.sum() == 0:
                joint_mutual_info = joint_predictive_entropy - joint_cond_entropy
                joint_mutual_info[selected_ind] = 0
                distance_cost_mask = area_cost > budget
                joint_mutual_info[distance_cost_mask] = 0

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
            selected_predicts = torch.einsum('ik,il->ikl', selected_predicts, stored_predicts.squeeze(0)).reshape(n_ensembles, -1).unsqueeze(0)
            stored_predicts = selected_predicts.clone()

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool, cost_total