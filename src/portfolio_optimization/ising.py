import numpy as np


# create a function that takes our qubo matrix and returns the coefficients of the Ising Hamiltonian
# generalised to any offset, here offset will be = AK^2
def qubo_to_ising(Q: np.ndarray, offset: float=0.0):
    n = len(Q)

    # initialize, J is a matric, h is a vector and c is a constant
    J = np.zeros((n,n))
    h = np.zeros(n)
    c = offset

    # implement h_i = -0.5 sum_j Q_ij
    for i in range(n):
        h[i] = -0.5*np.sum(Q[i])

    # implement J = 0.5*Q_ij, leave the diagonal and lower triangle at 0
    for i in range(n):
        for j in range(i+1,n):
            J[i,j]=0.5*Q[i,j]

    
    for i in range(n):
        c+= 0.5*Q[i,i]

        for j in range(i+1, n):
            c+=0.5*Q[i,j]
            
    return(h,J, c)