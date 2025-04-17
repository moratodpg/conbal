# models/bnn_pyro.py
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
# import arviz as az

class BayesianNNTrainer:
    def __init__(self, input_size, hidden_size, num_classes, device='cpu',
                 num_samples=500, warmup_steps=200, num_chains=1):
        """
        :param input_size: Number of input features (e.g., 384)
        :param hidden_size: Number of hidden units in layer
        :param num_classes: Number of output classes
        :param device: 'cpu' or 'cuda'
        :param num_samples: Number of MCMC samples to draw
        :param warmup_steps: Number of warmup (burn-in) steps
        :param num_chains: Number of parallel MCMC chains
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.num_chains = num_chains
        self.posterior_samples = None  # Will store MCMC samples

    def model(self, X, Y=None):
        """
        Define a 2-layer Bayesian neural network in Pyro with Normal priors.
        """
        # Weight/bias for layer 1 (input -> hidden)
        w1 = pyro.sample("w1",
                         dist.Normal(torch.zeros(self.hidden_size, self.input_size, device=self.device),
                                     torch.ones(self.hidden_size, self.input_size, device=self.device)))
        b1 = pyro.sample("b1",
                         dist.Normal(torch.zeros(self.hidden_size, device=self.device),
                                     torch.ones(self.hidden_size, device=self.device)))

        # Weight/bias for layer 2 (hidden -> output)
        w2 = pyro.sample("w2",
                         dist.Normal(torch.zeros(self.num_classes, self.hidden_size, device=self.device),
                                     torch.ones(self.num_classes, self.hidden_size, device=self.device)))
        b2 = pyro.sample("b2",
                         dist.Normal(torch.zeros(self.num_classes, device=self.device),
                                     torch.ones(self.num_classes, device=self.device)))

        # Forward pass
        hidden = torch.relu(X.matmul(w1.t()) + b1)  # shape: [batch, hidden_size]
        logits = hidden.matmul(w2.t()) + b2         # shape: [batch, num_classes]

        # Likelihood: multi-class classification
        with pyro.plate("data", X.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=Y)

    def train(self, train_loader):
        """
        Perform MCMC on the entire training set.  (Often you might combine
        all training data into a single batch for MCMC, but we'll show it
        with a loader below.)
        """
        # Collect all X and Y from train_loader into a single batch
        all_x = []
        all_y = []
        for X_batch, Y_batch, _ in train_loader:
            all_x.append(X_batch.to(self.device))
            all_y.append(Y_batch.to(self.device))

        X_full = torch.cat(all_x, dim=0)
        Y_full = torch.cat(all_y, dim=0)

        # Define NUTS kernel
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel,
                    num_samples=self.num_samples,
                    warmup_steps=self.warmup_steps,
                    num_chains=self.num_chains,
                    mp_context="spawn")  # On Windows/macOS can require 'spawn' "fork" is default on Linux
                    # progress_bar=False  # disable verbose output

        mcmc.run(X_full, Y_full)
        self.posterior_samples = mcmc.get_samples()

        # # Convert Pyro samples -> ArviZ InferenceData
        # idata = az.from_pyro(mcmc)  # or directly from samples: az.from_dict(samples)
        # # Print summary stats
        # print(az.summary(idata, round_to=2))  # mean, sd, hdi, ess, r_hat, etc.

    def predict_proba(self, X, n_samples=50):
        """
        Predict the probability distribution over classes by averaging
        over some subset of posterior samples.
        :param X: Input tensor of shape [batch, input_size]
        :param n_samples: How many posterior samples to use
        :return: Tensor of shape [batch, num_classes], representing average predicted probabilities.
        """
        if self.posterior_samples is None:
            raise RuntimeError("No posterior samples found. Have you called .train() yet?")

        # Get parameter sets
        # They are in a dict: e.g. {'w1': [num_samples, ...], 'b1': [...], 'w2': [...], 'b2': [...] }
        w1_samples = self.posterior_samples["w1"]  # shape [num_samples, hidden_size, input_size]
        b1_samples = self.posterior_samples["b1"]  # shape [num_samples, hidden_size]
        w2_samples = self.posterior_samples["w2"]  # shape [num_samples, num_classes, hidden_size]
        b2_samples = self.posterior_samples["b2"]  # shape [num_samples, num_classes]

        # If we want to do predictive checks with fewer than total samples:
        total_mcmc_samples = w1_samples.shape[0]
        chosen_idxs = torch.randperm(total_mcmc_samples)[:n_samples]

        # For each chosen sample, compute logits -> probabilities
        pred_probs = []
        for idx in chosen_idxs:
            w1 = w1_samples[idx]
            b1 = b1_samples[idx]
            w2 = w2_samples[idx]
            b2 = b2_samples[idx]

            hidden = F.relu(X.matmul(w1.t()) + b1)
            logits = hidden.matmul(w2.t()) + b2
            probs = F.softmax(logits, dim=-1)  # [batch, num_classes]
            pred_probs.append(probs)

        # Average the probabilities across the posterior samples
        avg_probs = torch.stack(pred_probs, dim=0).mean(dim=0)  # shape [batch, num_classes]
        return avg_probs

    def evaluate(self, test_loader, n_samples=50):
        """
        Evaluate accuracy on the test set by averaging across posterior samples.
        Returns classification accuracy (float).
        """
        correct = 0
        total = 0

        self.eval_mode = True
        for X_batch, Y_batch, _ in test_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            probs = self.predict_proba(X_batch, n_samples=n_samples)
            preds = probs.argmax(dim=-1)
            correct += (preds == Y_batch).sum().item()
            total += Y_batch.size(0)

        return correct / total
    
    def predict_posterior_samples(self, X):
        """
        """
        if self.posterior_samples is None:
            raise RuntimeError("No posterior samples found. Have you called .train() yet?")

        num_samples = self.posterior_samples["w1"].shape[0]  # Number of posterior draws
        num_test = X.shape[0]  # Number of test samples

        predictions = torch.zeros((num_test, num_samples, self.num_classes))  # Storage for predictions

        # Iterate over MCMC posterior samples
        for i in range(num_samples):
            # Sampled weights and biases
            w1 = self.posterior_samples["w1"][i]
            b1 = self.posterior_samples["b1"][i]
            w2 = self.posterior_samples["w2"][i]
            b2 = self.posterior_samples["b2"][i]

            # Forward pass with ReLU
            hidden = F.relu(X.matmul(w1.t()) + b1)
            logits = hidden.matmul(w2.t()) + b2

            # Convert logits to class probabilities
            probs = F.softmax(logits, dim=-1)  # [batch, num_classes]

            # Store predictions
            predictions[:, i, :] = probs

        return predictions