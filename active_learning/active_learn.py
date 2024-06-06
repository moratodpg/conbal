import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActiveLearning:
    def __init__(self, num_active_points):
        self.num_active_points = num_active_points

    def get_active_points(self, net_current, num_forwards, buildings_dataset, idx_pool):
        loss_iterations = []
        predictions = []
        criterion_item = torch.nn.CrossEntropyLoss(reduction='none')
        net_current.train()
        for i in range(num_forwards):
            with torch.no_grad():
                predicted = net_current(buildings_dataset.input_tensor[idx_pool][:, :-22])
                # print(predicted.shape, predicted)

            target = buildings_dataset.output_tensor[idx_pool]

            loss = criterion_item(predicted, target)
            predicted = F.softmax(predicted, dim=1)
            loss_iterations.append(loss)
            predictions.append(predicted)

        ## Dimensions: samples x iterations
        losses = torch.stack(loss_iterations, dim=1)

        ## Dimensions: samples x iterations x classes
        predicts = torch.stack(predictions, dim=1)
        # print(predicts.shape)

        # Compute predictive entropy
        avg_predicts = torch.mean(predicts, dim=1)
        eps = 1e-9
        avg_probs_clamped = torch.clamp(avg_predicts, min=eps)
        entropy = -torch.sum(avg_probs_clamped * torch.log2(avg_probs_clamped), dim=1)

        # Compute expected entropy
        prob_clamped = torch.clamp(predicts, min=eps)
        entropy_i = -torch.sum(prob_clamped * torch.log2(prob_clamped), dim=2)
        entropy_sum = entropy_i.sum(dim=1) / num_forwards

        # Compute mutual information
        mutual_info = entropy - entropy_sum

        # active points
        n_active_points = self.num_active_points
        mi_values, mi_indices = mutual_info.topk(n_active_points)
        selected_ind = mi_indices.tolist()

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]

        return selected_idx_pool
    
    def get_random_points(self, idx_pool, loc_identifier):
        # create a list of random numbers as integers from 0 to len(idx_pool)
        camp_cost = 0.03
        ind_cost = 1
        
        n_active_points = self.num_active_points
        random_idx_pool = np.random.choice(idx_pool, n_active_points, replace=False)
        random_idx_pool = random_idx_pool.tolist()

        selected_locations = loc_identifier[random_idx_pool]
        regions_num = torch.unique(selected_locations).shape[0]
        cost_accum = camp_cost * regions_num + ind_cost * (n_active_points - 1)
        return random_idx_pool, cost_accum
    
    def get_multiple_active_points(self, net_current, num_forwards, buildings_dataset, idx_pool):
        loss_iterations = []
        predictions = []
        criterion_item = torch.nn.CrossEntropyLoss(reduction='none')
        net_current.train()
        for i in range(num_forwards):
            with torch.no_grad():
                predicted = net_current(buildings_dataset.input_tensor[idx_pool][:, :-22])
                # print(predicted.shape, predicted)

            target = buildings_dataset.output_tensor[idx_pool]

            loss = criterion_item(predicted, target)
            predicted = F.softmax(predicted, dim=1)
            loss_iterations.append(loss)
            predictions.append(predicted)

        ## Dimensions: samples x iterations
        losses = torch.stack(loss_iterations, dim=1)

        ## Dimensions: samples x iterations x classes
        predicts = torch.stack(predictions, dim=1)
        # print(predicts.shape)

        # Compute predictive entropy
        avg_predicts = torch.mean(predicts, dim=1)
        eps = 1e-9
        avg_probs_clamped = torch.clamp(avg_predicts, min=eps)
        entropy = -torch.sum(avg_probs_clamped * torch.log2(avg_probs_clamped), dim=1)

        # Compute expected entropy
        prob_clamped = torch.clamp(predicts, min=eps)
        entropy_i = -torch.sum(prob_clamped * torch.log2(prob_clamped), dim=2)
        entropy_sum = entropy_i.sum(dim=1) / num_forwards

        # Compute mutual information
        mutual_info = entropy - entropy_sum

        # active points
        n_active_points = self.num_active_points
        mi_values, mi_indices = mutual_info.topk(1)
        selected_ind = mi_indices.tolist()

        for _ in range(n_active_points - 1):

            # Compute the conditional entropy
            sum_conditional_entropy = torch.zeros(1)
            for index in selected_ind: # Iterate over already selected buildings
                sum_conditional_entropy += entropy_sum[index]
            joint_cond_entropy = entropy_sum + sum_conditional_entropy
            # print(sum_conditional_entropy, joint_cond_entropy, entropy_sum)

            # Compute joint entropy
            ## Expanded joint entropy for all possible combinations
            tensor_list = []
            tensor_list.append(predicts)
            for index in selected_ind:
                tensor_list.append(predicts[index, :, :].unsqueeze(0))
            expanded_joint_entropy = self.combine_class_products(tensor_list)
            #print(expanded_joint_entropy.shape, expanded_joint_entropy)
            avg_combinedj_entropy = torch.mean(expanded_joint_entropy, dim=1)
            eps = 1e-9
            avg_combinedj_clamped = torch.clamp(avg_combinedj_entropy, min=eps)
            joint_entropy = -torch.sum(avg_combinedj_clamped * torch.log2(avg_combinedj_clamped), dim=1)

            # Joint mutual information
            joint_mutual_info = joint_entropy - joint_cond_entropy

            # Mask already selected indices
            joint_mutual_info[selected_ind] = 0

            # Select the next batch active point
            mi_values, mi_indices = joint_mutual_info.topk(1)
            selected_ind.append(mi_indices.item())

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]

        return selected_idx_pool
    
    def get_active_points_cost(self, net_current, num_forwards, buildings_dataset, idx_pool, loc_identifier):
        camp_cost = 0.5 #(Before it was 0.03, then 0.1)
        ind_cost = 1
        loss_iterations = []
        predictions = []
        criterion_item = torch.nn.CrossEntropyLoss(reduction='none')
        net_current.train()
        for i in range(num_forwards):
            with torch.no_grad():
                predicted = net_current(buildings_dataset.input_tensor[idx_pool][:, :-22])
                # print(predicted.shape, predicted)

            target = buildings_dataset.output_tensor[idx_pool]

            loss = criterion_item(predicted, target)
            predicted = F.softmax(predicted, dim=1)
            loss_iterations.append(loss)
            predictions.append(predicted)

        ## Dimensions: samples x iterations
        losses = torch.stack(loss_iterations, dim=1)

        ## Dimensions: samples x iterations x classes
        predicts = torch.stack(predictions, dim=1)
        # print(predicts.shape)

        # Compute predictive entropy
        avg_predicts = torch.mean(predicts, dim=1)
        eps = 1e-9
        avg_probs_clamped = torch.clamp(avg_predicts, min=eps)
        entropy = -torch.sum(avg_probs_clamped * torch.log2(avg_probs_clamped), dim=1)

        # Compute expected entropy
        prob_clamped = torch.clamp(predicts, min=eps)
        entropy_i = -torch.sum(prob_clamped * torch.log2(prob_clamped), dim=2)
        entropy_sum = entropy_i.sum(dim=1) / num_forwards

        # Compute mutual information
        mutual_info = entropy - entropy_sum

        # active points
        n_active_points = self.num_active_points
        mi_values, mi_indices = mutual_info.topk(1)

        selected_ind = mi_indices.tolist()
        # Auditing locations
        build_locations = loc_identifier[idx_pool]
        selected_locations = build_locations[mi_indices]
        cost_accum = camp_cost

        for _ in range(n_active_points - 1):

            # Compute the conditional entropy
            sum_conditional_entropy = torch.zeros(1)
            for index in selected_ind: # Iterate over already selected buildings
                sum_conditional_entropy += entropy_sum[index]
            joint_cond_entropy = entropy_sum + sum_conditional_entropy
            # print(sum_conditional_entropy, joint_cond_entropy, entropy_sum)

            # Compute joint entropy
            ## Expanded joint entropy for all possible combinations
            tensor_list = []
            tensor_list.append(predicts)
            for index in selected_ind:
                tensor_list.append(predicts[index, :, :].unsqueeze(0))
            expanded_joint_entropy = self.combine_class_products(tensor_list)
            #print(expanded_joint_entropy.shape, expanded_joint_entropy)
            avg_combinedj_entropy = torch.mean(expanded_joint_entropy, dim=1)
            eps = 1e-9
            avg_combinedj_clamped = torch.clamp(avg_combinedj_entropy, min=eps)
            joint_entropy = -torch.sum(avg_combinedj_clamped * torch.log2(avg_combinedj_clamped), dim=1)

            # Joint mutual information
            joint_mutual_info = joint_entropy - joint_cond_entropy

            # Cost tensor based on location
            cost_tensor = torch.ones(len(idx_pool)) * (camp_cost + ind_cost)
            inspected_build = torch.isin(build_locations, selected_locations)
            cost_tensor[inspected_build] = ind_cost

            # MI divided by cost
            joint_mutual_info = joint_mutual_info / cost_tensor

            # Mask already selected indices
            joint_mutual_info[selected_ind] = 0

            # Select the next batch active point
            mi_values, mi_indices = joint_mutual_info.topk(1)
            selected_locations = torch.cat((selected_locations, build_locations[mi_indices]))
            selected_ind.append(mi_indices.item())
            cost_accum += cost_tensor[mi_indices].item()

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]

        return selected_idx_pool, cost_accum
    
    def combine_class_products(self, tensors):
        # Number of tensors
        n_tensors = len(tensors)
        
        # Calculate the total number of class combinations
        total_classes = 2 ** n_tensors
        
        # Initial combination tensor
        samples, iterations, _ = tensors[0].shape
        combination_tensor = torch.ones(samples, iterations, total_classes)
        
        # Iterate through each class combination
        for i in range(total_classes):
            # Compute the index for each tensor's class (0 or 1) based on the combination
            indices = [(i >> j) & 1 for j in range(n_tensors)]
            
            # Compute the product for the current combination
            for tensor_idx, class_idx in enumerate(indices):
                # Select the class index for the current tensor and multiply
                combination_tensor[:, :, i] *= tensors[tensor_idx][:, :, class_idx]
        
        return combination_tensor
        
         
