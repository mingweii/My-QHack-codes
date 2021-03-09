#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np
from pennylane import numpy as npx
def variational_ansatz(params, wires):
    """The variational ansatz circuit.

    Fill in the details of your ansatz between the # QHACK # comment markers. Your
    ansatz should produce an n-qubit state of the form

        a_0 |10...0> + a_1 |01..0> + ... + a_{n-2} |00...10> + a_{n-1} |00...01>

    where {a_i} are real-valued coefficients.

    Args:
         params (np.array): The variational parameters.
         wires (qml.Wires): The device wires that this circuit will run on.
    """

    # QHACK #
    N=len(wires)

    qml.PauliX(wires=0)

    for i, param in enumerate(params):
        qml.RY(-param, wires=i+1)
        qml.CZ(wires=[i+1,i])
        qml.RY(param,wires=i+1)

    # Reversed CNOTs
    for i in range(len(params)):
        qml.Hadamard(wires=i)
        qml.Hadamard(wires=i+1)
        qml.CNOT(wires=[i,i+1])
        qml.Hadamard(wires=i)
        qml.Hadamard(wires=i+1)


    #qml.QubitStateVector(npx.array(params,requires_grad=False),wires=wires)
    # QHACK #


def run_vqe(H):
    """Runs the variational quantum eigensolver on the problem Hamiltonian using the
    variational ansatz specified above.

    Fill in the missing parts between the # QHACK # markers below to run the VQE.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The ground state energy of the Hamiltonian.
    """
    energy = 0

    # QHACK #

    # Initialize the quantum device
    dev=qml.device('default.qubit',wires=H.wires)

    # Randomly choose initial parameters (how many do you need?)
    num_qubits = len(H.wires)

    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=num_qubits-1)

    # Set up a cost function

    cost_fn=qml.ExpvalCost(variational_ansatz,H,dev)

    # Set up an optimizer
    eta=0.2
    opt = qml.AdamOptimizer(stepsize=eta)
    max_iterations= 500
    # Run the VQE by iterating over many steps of the optimizer
    conv_tol=1e-6

    for n in range(max_iterations):
        #print(params)
        params, prev_energy = opt.step_and_cost(cost_fn,params)
        #print(params)
        energy = cost_fn((params))
        conv= np.abs(energy - prev_energy)
        #if n % 20 == 0:
        #print('Iteration = {:},  Energy = {:.8f}, Convergence={:.8f} Ha'.format(n, energy,conv))
        if 1e-4<=conv<1e-2:
            opt.update_stepsize(0.05)
        elif conv_tol<conv<1e-4:
            opt.update_stepsize(0.01)
        elif conv <= conv_tol:
            break
        else:
            continue

    #print()
    #print('Final convergence parameter = {:.8f} Ha'.format(conv))
    #print('Final value of the ground-state energy = {:.8f} Ha'.format(energy))
    #print('Accuracy with respect to the FCI energy: {:.8f} Ha ({:.8f} kcal/mol)'.format(
    #np.abs(energy - (-1.136189454088)), np.abs(energy - (-1.136189454088))*627.503
    #))
    #print()
    #print('Final circuit parameters = \n', params)
    # QHACK #

    # Return the ground state energy
    return energy


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    ground_state_energy = run_vqe(H)
    print(f"{ground_state_energy:.6f}")
