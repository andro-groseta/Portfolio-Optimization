import numpy as np
from portfolio_optimization.markowitz import mvp_analytic

import torch
import torch.nn as nn


# generate a random m, sigma, and m_p
def generate_portfolio_instance(n_assets=4, seed = None):
    # create random number generator
    rng = np.random.default_rng(seed)

    # generate expected returns m_i, sampled from range 2% to 14%
    m = rng.uniform(0.02, 0.14, n_assets)

    # generate random normally distributed matrix B
    B = rng.normal(loc=0, scale = 0.05, size=(n_assets, n_assets))

    # create positive definit matrix Sigma
    k = rng.uniform(0.0001, 0.001)
    sigma = B.T @ B + k*np.eye(n_assets)

    # target return
    m_p = rng.uniform(np.min(m), np.max(m))

    return m, sigma, m_p


# generate a whole dataset of portfolios
def generate_dataset(n_samples=5000, n_assets=4, seed=42):

    rng = np.random.default_rng(seed)

    X = []
    Y = []

    for k in range(n_samples):

        # Generate a new random portfolio instance
        instance_seed = int(rng.integers(0, 2**32))
        m, sigma, m_p = generate_portfolio_instance(n_assets, instance_seed)

        # Calculate its exact optimal weights
        pi_star = mvp_analytic(sigma, m, m_p)

        # Convert its inputs into a feature vector and sigma into only top triangle
        sigma_features = sigma[np.triu_indices(n_assets)]
        features = np.concatenate([m, sigma_features, [m_p]])

        # Store the features in X and optimal weights in Y
        X.append(features)
        Y.append(pi_star)

    return np.array(X), np.array(Y)





# initialize subclass of nn.Module
class PortfolioMLP(nn.Module):

    def __init__(self, input_dim=15, hidden_dim=64, output_dim=4):
        # Initialize the parent nn.Module class
        super().__init__()

        # Define the three fully connected layers
        # Input: 15 portfolio features -> 64 neurons
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # First hidden layer: 64 -> 64 neurons
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Second hidden layer: 64 -> 4 predicted portfolio weights
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # ReLU activation introduces nonlinearity
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass the inputs through the first layer and apply ReLU
        x = self.relu(self.fc1(x))

        # Pass through the second hidden layer and apply ReLU
        x = self.relu(self.fc2(x))

        # Output the predicted portfolio weights without activation
        # Negative weights are allowed in our Markowitz formulation
        x = self.fc3(x)

        return x



def constrained_loss(predictions, targets, m, m_p, lambda1=1, lambda2=100):

    # MSE between predictions and targets
    mse_loss = torch.mean((predictions -targets)**2)

    # budget constraint penalty
    budget_residual = predictions.sum(dim=1) -1
    budget_loss = torch.mean(budget_residual**2)

    # target-return constraint penalty
    predicted_returns = torch.sum(m*predictions, dim=1)
    return_residual = predicted_returns-m_p
    return_loss = torch.mean(return_residual**2)

    # combine
    total_loss = mse_loss +lambda1*budget_loss + lambda2*return_loss

    return total_loss