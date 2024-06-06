from torch.utils.data import Dataset
import torch

class Buildings(Dataset):
    def __init__(self, input_tensor, output_tensor, transform=None):
        """
        Args:
            input_tensor (torch.Tensor): Input tensor.
            output_tensor (torch.Tensor): Output tensor.
            train_ratio (float): Ratio of the dataset to be used for training.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert len(output_tensor) == len(input_tensor), "Output tensor must have the same length as input tensors"

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.transform = transform

    def __len__(self):
        # Return the length of the training set if the dataset is used for training, 
        # otherwise return the length of the testing set
        return len(self.output_tensor) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract samples from input tensors and output tensor
        input_samples = self.input_tensor[idx]
        output_sample = self.output_tensor[idx]

        # Applying any specified transformations
        # if self.transform:
        #     input_samples, output_sample = self.transform(input_samples, output_sample)

        return input_samples, output_sample, idx