import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


# create the circuit discussed in the notebook
# input params should be of the form [theta1, theta2,..., beta1, beta2,...]
def qaoa_circuit(h ,J , params):
    # slice params in two, to give thetas and betas
    p = len(params) // 2
    thetas = params[:p]
    betas = params[p:]

    # initialize quantumcircuit with the same amount of qubits as assets
    n = len(h)
    qc = QuantumCircuit(n)

    # if the number of thetas and betas is unequal we have a problem
    p = len(thetas)
    if len(betas) !=p:
        raise ValueError("thetas and betas have to have same length")

    # inital Haddamard gates
    for i in range(n):
        qc.h(i)

    for layer in range(p):
        # first do rz for h
        for i in range(n):
            qc.rz(2*thetas[layer]*h[i], i)
        # then do rzz for J
        for i in range(n):
            for j in range(i+1, n):
                qc.rzz(2*thetas[layer]*J[i,j], i, j)

        # RX gate for mixer
        for i in range(n):
            qc.rx(2*betas[layer], i)

    return(qc)


# finding the expected energy of a state after evolution
def qaoa_expected_energy(params, h, J, energies):

    # initialize QC
    qc = qaoa_circuit(h, J, params)

    # create a statevector after going through circuit and save its probabilities
    psi = Statevector.from_instruction(qc)
    probabilities = psi.probabilities()

    return energies @ probabilities


# numerical optimization of the expected energy
def optimize_qaoa(h, J, energies, initial_params):

    result = minimize( fun=qaoa_expected_energy, x0=initial_params, args=(h, J, energies), method="COBYLA", 
                     options={"maxiter": 3000, "tol": 1e-8})

    return result