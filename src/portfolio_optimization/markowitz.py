import numpy as np
from scipy.optimize import minimize

#### pi = lambda sigma^-1 m + mu sigma^-1 e

def mvp_analytic(sigma: np.ndarray, m: np.ndarray, m_p: float):

   # number of assets
   n= len(m)
   # vector of 1s
   e = np.ones(n)

   # x = sigma^-1 m, y = sigma^-1 e
   x = np.linalg.solve(sigma, m)
   y = np.linalg.solve(sigma, e)

   # B (lambda, mu)^T = (m_p, 1)^T, B=([m^T x, m^T y], [e^T x, e^T y])   --->   construct B
   B = np.array([ 
      [m.T @ x, m.T @ y],
      [e.T @ x, e.T @ y]])

   # a = (m_p, 1)^T      and     lagrange = (lambda, mu)^T
   a = np.array([m_p, 1])
   lagrange = np.linalg.solve(B, a)

   pi= lagrange[0] * x + lagrange[1] * y

   return(pi)



def mvp_numerical(sigma: np.ndarray, m: np.ndarray, m_p: float, allow_short: bool):

   n = len(m)
   e = np.ones(n)

   # define the function f(pi) = 1/2 pi^T sigma pi
   def objective(pi):
      return(0.5 * pi @ sigma @ pi)

   # define our constraints: 0 = pi^T m -m_p and 0 = pi^T e -1
   constraints = [{"type": "eq","fun": lambda pi: pi @ m -m_p}, 
                  {"type": "eq", "fun": lambda pi: pi @ e -1}]

   # We need to put in an initial guess from which the minimizer starts --->  equally weighted portfolio
   initial_guess = np.array([1/n]*n) 

   # Bounds for short-selling
   if allow_short:
      bounds = [(None, None)] * n
   else:
      bounds = [(0, None)] * n

   # Use the minimize function
   result = minimize(fun = objective, x0 = initial_guess, method = "SLSQP", constraints = constraints,
                      options = {"ftol": 1e-12, "maxiter": 1000}, bounds = bounds)

   return(result)




# pi_GMVP = (sigma^-1 e)/(e^T sigma^-1 e)

def gmvp(sigma: np.ndarray):
   # number of assets is going to be length of the covariance matrix
   n = len(sigma)
   e = np.ones(n)

   #sigma x = e
   x = np.linalg.solve(sigma, e)

   denom = e.T @ x

   return(x/denom)