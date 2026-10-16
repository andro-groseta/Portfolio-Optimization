# Portfolio Optimization: Classical and Quantum Approaches

This project explores portfolio optimization using classical mathematical methods, quantum optimization algorithms, and a supervised machine learning approach. It started from the Markowitz portfolio optimization problem introduced in my Financial Mathematics course, which I wanted to investigate further through numerical implementations and alternative optimization approaches.

The project currently consists of three notebooks. The first focuses on the classical mean-variance optimization problem, while the second extends the problem to discrete portfolio selection and investigates its solution using the Quantum Approximate Optimization Algorithm (QAOA). In the third I ask the question, whether a learned model can approximate solutions across varying portfolio parameters.

## 1. Classical Markowitz Portfolio Optimization

The first notebook implements the classical Markowitz mean-variance optimization problem:

$$\min_{\pi}\frac{1}{2}\pi^T\Sigma\pi$$

subject to
$$\pi^T\mathbf{1}=1,\qquad \pi^Tm=m_p$$

where $\pi$ represents the portfolio weights, $\Sigma$ the covariance matrix of asset returns, $m$ the expected returns, and $m_p$ the target portfolio return. The problem is solved analytically using Lagrange multipliers and numerically using SciPy's constrained optimization methods. Both solutions are compared to verify the implementation. The notebook also investigates the global minimum-variance portfolio and constructs the efficient frontier, including the effect of introducing no-short-selling constraints.

## 2. Discrete Portfolio Optimization and QAOA

The second notebook extends the portfolio problem to discrete asset selection. Binary variables $z_i\in\{0,1\}$ indicate whether an asset is selected, with the constraint that exactly K assets must be included. The optimization objective combines portfolio risk and expected return:

$$ C(z)=\frac{\gamma}{2K^2}z^T\Sigma z-\frac{1}{K}m^Tz $$

To incorporate the cardinality constraint, we introduce the penalty term

$$ A\left(\sum_i z_i-K\right)^2 $$

resulting in a Quadratic Unconstrained Binary Optimization (QUBO) problem. The QUBO formulation is then mapped to an Ising Hamiltonian,

$$ H_C = cI + \sum_i h_i Z_i + \sum_{i<j}J_{ij}Z_iZ_j $$

which can be implemented using quantum gates. The notebook implements QAOA using Qiskit and optimizes its variational parameters using SciPy. An exhaustive classical search provides the exact optimum against which the quantum algorithm is evaluated.

### Numerical results

For the four-asset example, we investigated the influence of the penalty parameter and circuit depth on QAOA's performance. Increasing the circuit depth from p=1 to p=2 improved the expected energy and increased the probability of measuring a feasible portfolio from approximately 73.8% to 99.9%.

However, the probability of obtaining the optimal portfolio only increased from approximately 12.5% to 16.9%. This illustrates that minimizing the expected energy and satisfying the constraints do not necessarily imply a high probability of measuring the optimal solution. The p=3 optimization did not converge within the chosen evaluation limit, making further conclusions about additional circuit depth difficult.

Since the problem involves only four assets and is simulated using exact statevectors, the results do not demonstrate any quantum computational advantage. The aim is instead to understand the mathematical connection between classical optimization, QUBO formulations, Ising Hamiltonians, and variational quantum algorithms.

## 3. Machine Learning for Portfolio Optimization

The third notebook investigates whether a neural network can approximate the analytical Markowitz solution. A synthetic dataset is generated using the analytical solutions as training labels, and a multilayer perceptron (MLP) is implemented in PyTorch to predict the optimal portfolio weights.

We compare a standard network trained using mean squared error (MSE) with a constraint-aware network that additionally penalizes violations of the budget and target-return constraints. The effect of different penalty coefficients is also investigated. On the test dataset, the constrained network reduced the mean budget violation by approximately 33% and the return violation by 40%, although its prediction MSE increased. Both networks learned the general relationship between the portfolio inputs and optimal weights but struggled with extreme allocations.

The results illustrate how incorporating mathematical constraints into the loss function can improve feasibility without guaranteeing exact constraint satisfaction.

## 4. Project Structure

```text
Portfolio Optimization/
├── notebooks/
│   ├── 01_Markowitz.ipynb
│   ├── 02_Discrete_Portoflio_Optimization.ipynb
|   └── 03_ML_portfolio_optimization.ipynb
├── src/
│   └── portfolio_optimization/
│       ├── __init__.py
│       ├── discrete.py
│       ├── ising.py
|       ├── learning.py
│       ├── markowitz.py
│       └── qaoa.py
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

The `src/portfolio_optimization/` directory contains the reusable implementations, while the notebooks explain the mathematical derivations, numerical experiments, and results.

## 5. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/andro-groseta/Portfolio-Optimization.git
cd Portfolio-Optimization
python -m pip install -r requirements.txt
```

The notebooks can be run using Jupyter or VS Code with a Python environment containing the required dependencies.

## 6. Outlook

Possible extensions include investigating larger portfolio instances and alternative QAOA mixers that preserve the cardinality constraint. It would be interesting to see how QAOA performs as the number of assets increases.

For the machine learning approach, a possible next step would be to introduce a projection layer that ensures the budget and target-return constraints are satisfied exactly. Different network architectures could also be investigated to see whether they improve the prediction of extreme portfolio weights.