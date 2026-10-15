import numpy as np

def discrete_portfolio_cost(z: np.ndarray, sigma:np.ndarray,m:np.ndarray, K:int, gamma:float):
    return(0.5*gamma* z.T @ sigma @ z/(K**2) - m.T @z/K)

# Qii = gamma/2K^2 Sigma_ii - m_i/K +A(1-2K)
# Q_ij = gamma/2k^2 Sigma_ij +A
def QUBO_matrix(sigma:np.array, m:np.array, K:int, gamma:float, A:float):
    n = len(m)
    e = np.ones(n)

    #initiate a matrix of zeroes and assign each entry
    Q = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                Q[i,j] = gamma/(2*K**2)*sigma[i][j] -m[i]/K +A*(1-2*K)
            else:
                Q[i,j] = gamma/(2*K**2)*sigma[i][j] +A
    return(Q)



def cost_matrix(z:np.array, Q:np.array, K:float, A:float):
    return  z @ Q @ z + A * K**2

def cost_direct( z, sigma, m, K, gamma, A):
    return discrete_portfolio_cost(z, sigma, m, K, gamma) + A * (np.sum(z) - K)**2