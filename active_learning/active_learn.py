import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActiveLearning:
    def __init__(self, num_active_points):
        self.num_active_points = num_active_points

    def get_active_points(self, net_current, num_forwards, buildings_dataset, idx_pool):
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        mutual_info = self.mutual_information(predicts)
        # active points
        _, mi_indices = mutual_info.topk(self.num_active_points)
        selected_ind = mi_indices.tolist()
        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]
        return selected_idx_pool
    
    # def get_random_points(self, idx_pool, loc_identifier):
    #     # create a list of random numbers as integers from 0 to len(idx_pool)
    #     camp_cost = 0.03
    #     ind_cost = 1
        
    #     n_active_points = self.num_active_points
    #     random_idx_pool = np.random.choice(idx_pool, n_active_points, replace=False)
    #     random_idx_pool = random_idx_pool.tolist()

    #     selected_locations = loc_identifier[random_idx_pool]
    #     regions_num = torch.unique(selected_locations).shape[0]
    #     cost_accum = camp_cost * regions_num + ind_cost * (n_active_points - 1)
    #     return random_idx_pool, cost_accum

    def get_random_points(self, idx_pool):
        # create a list of random numbers as integers from 0 to len(idx_pool)
        random_idx_pool = np.random.choice(idx_pool, self.num_active_points, replace=False)
        random_idx_pool = random_idx_pool.tolist()
        return random_idx_pool
    
    def get_multiple_active_points(self, net_current, num_forwards, buildings_dataset, idx_pool):
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)
        entropy = self.predictive_entropy(predicts)
        entropy_sum = self.expected_conditional_entropy(predicts)
        mutual_info = entropy - entropy_sum

        # Get first active point
        n_active_points = self.num_active_points
        _, mi_indices = mutual_info.topk(1)
        selected_ind = mi_indices.tolist()

        # Evaluate the first point and store it
        selected_predicts = predicts[selected_ind[-1]].unsqueeze(0)
        stored_predicts = selected_predicts.clone()

        for _ in range(n_active_points - 1):

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

            # Mask already selected indices
            joint_mutual_info[selected_ind] = 0

            # Select the next batch active point
            _, mi_indices = joint_mutual_info.topk(1)
            selected_ind.append(mi_indices.item())

            # Store the selected point
            selected_predicts = predicts[selected_ind[-1]]
            selected_predicts = torch.einsum('ik,il->ikl', selected_predicts, stored_predicts.squeeze(0)).reshape(num_forwards, -1).unsqueeze(0)
            stored_predicts = selected_predicts.clone()

        # From the selected indices, get the indices from the pool
        selected_idx_pool = [idx_pool[i] for i in selected_ind]

        return selected_idx_pool
    
    def get_active_points_cost(self, net_current, num_forwards, buildings_dataset, idx_pool, loc_identifier):
        camp_cost = 0.5 #(Before it was 0.03, then 0.1)
        ind_cost = 1
        predicts = self.predict(net_current, buildings_dataset.input_tensor[idx_pool], num_forwards)

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
        
        # Initial combination tensor
        samples, iterations, num_classes = tensors[0].shape
        # Calculate the total number of class combinations
        total_classes = num_classes ** n_tensors
        combination_tensor = torch.ones(samples, iterations, total_classes)
        
        # Iterate through each class combination
        for i in range(total_classes):
            # Compute the index for each tensor's class (0 or 1) based on the combination
            ##Binary
            # indices = [(i >> j) & 1 for j in range(n_tensors)]
            ## Multi-class
            indices = []
            temp = i
            for _ in range(n_tensors):
                indices.append(temp % num_classes)
                temp //= num_classes

            # Compute the product for the current combination
            for tensor_idx, class_idx in enumerate(indices):
                # Select the class index for the current tensor and multiply
                combination_tensor[:, :, i] *= tensors[tensor_idx][:, :, class_idx]
        
        return combination_tensor
    
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
