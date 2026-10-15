# Portfolio Optimization: Classical and Quantum Approaches

This project explores portfolio optimization using classical mathematical methods and quantum optimization algorithms. It started from the Markowitz portfolio optimization problem introduced in my Financial Mathematics course, which I wanted to investigate further through numerical implementations and alternative optimization approaches.

The project currently consists of two notebooks. The first focuses on the classical mean-variance optimization problem, while the second extends the problem to discrete portfolio selection and investigates its solution using the Quantum Approximate Optimization Algorithm (QAOA).

## 1. Classical Markowitz Portfolio Optimization

The first notebook implements the classical Markowitz mean-variance optimization problem:

$$\min_{\pi}\frac{1}{2}\pi^T\Sigma\pi$$

subject to
$$\pi^T\mathbf{1}=1,\qquad \pi^Tm=m_p$$

where $\pi$ represents the portfolio weights, $\Sigma$ the covariance matrix of asset returns, $m$ the expected returns, and $m_p$ the target portfolio return.

The problem is solved analytically using Lagrange multipliers and numerically using SciPy's constrained optimization methods. Both solutions are compared to verify the implementation.

The notebook also investigates the global minimum-variance portfolio and constructs the efficient frontier, including the effect of introducing no-short-selling constraints.

## 2. Discrete Portfolio Optimization and QAOA

The second notebook extends the portfolio problem to discrete asset selection. Binary variables $z_i\in\{0,1\}$ indicate whether an asset is selected, with the constraint that exactly K assets must be included.

The optimization objective combines portfolio risk and expected return:

$$ C(z)=\frac{\gamma}{2K^2}z^T\Sigma z-\frac{1}{K}m^Tz $$

To incorporate the cardinality constraint, we introduce the penalty term

$$ A\left(\sum_i z_i-K\right)^2 $$

resulting in a Quadratic Unconstrained Binary Optimization (QUBO) problem.

The QUBO formulation is then mapped to an Ising Hamiltonian,

$$ H_C=cI+\sum_i h_iZ_i+\sum_{i<j}J_{ij}Z_iZ_j $$

which can be implemented using quantum gates.

The notebook implements QAOA using Qiskit and optimizes its variational parameters using SciPy. An exhaustive classical search provides the exact optimum against which the quantum algorithm is evaluated.

### Numerical results

For the four-asset example, we investigated the influence of the penalty parameter and circuit depth on QAOA's performance.

Increasing the circuit depth from p=1 to p=2 improved the expected energy and increased the probability of measuring a feasible portfolio from approximately 73.8% to 99.9%.

However, the probability of obtaining the optimal portfolio only increased from approximately 12.5% to 16.9%. This illustrates that minimizing the expected energy and satisfying the constraints do not necessarily imply a high probability of measuring the optimal solution.

The p=3 optimization did not converge within the chosen evaluation limit, making further conclusions about additional circuit depth difficult.

Since the problem involves only four assets and is simulated using exact statevectors, the results do not demonstrate any quantum computational advantage. The aim is instead to understand the mathematical connection between classical optimization, QUBO formulations, Ising Hamiltonians, and variational quantum algorithms.

## 3. Project Structure

```text
Portfolio Optimization/
├── notebooks/
│   ├── 01_classical_markowitz.ipynb
│   └── 02_discrete_qaoa.ipynb
├── src/
│   └── portfolio_optimization/
│       ├── __init__.py
│       ├── markowitz.py
│       ├── discrete.py
│       ├── ising.py
│       └── qaoa.py
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

The `src/portfolio_optimization/` directory contains the reusable implementations, while the notebooks explain the mathematical derivations, numerical experiments, and results.

## 4. Installation

Clone the repository and install the required dependencies:

git clone https://github.com/andro-groseta/Portfolio-Optimization
cd portfolio-optimization
python -m pip install -r requirements.txt

The notebooks can be run using Jupyter or VS Code with a Python environment containing the required dependencies.

## 5. Outlook

As a next step, I plan to investigate how machine learning methods can be applied to portfolio optimization. I would like to explore whether a neural network can be used to approximate or improve optimization procedures and compare its performance with the classical and quantum approaches implemented here.

Other possible extensions include investigating larger portfolio instances and alternative QAOA mixers that preserve the cardinality constraint.