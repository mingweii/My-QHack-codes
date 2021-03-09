#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np

def variational_ansatz(params, wires):
    """
    Hardware efficient ansatz taken from VQE 100.

    Args:
        params (np.ndarray): An array of floating-point numbers with size (n, 3),
            where n is the number of parameter sets required (this is determined by
            the problem Hamiltonian).
        wires (qml.Wires): The device wires this circuit will run on.
    """
    n_qubits = len(wires)
    n_rotations = len(params)

    if n_rotations > 1:
        n_layers = n_rotations // n_qubits
        n_extra_rots = n_rotations - n_layers * n_qubits

        # Alternating layers of unitary rotations on every qubit followed by a
        # ring cascade of CNOTs.
        for layer_idx in range(n_layers):
            layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
            qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
            qml.broadcast(qml.CNOT, wires, pattern="ring")

        # There may be "extra" parameter sets required for which it's not necessarily
        # to perform another full alternating cycle. Apply these to the qubits as needed.
        if n_extra_rots>0:
            extra_params = params[-n_extra_rots:, :]
            extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
            qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
    else:
        # For 1-qubit case, just a single rotation to the qubit
        qml.Rot(*params[0], wires=wires[0])

def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)
    # QHACK #
    import time
    clock=time.time()
    # Initialize parameters
    num_qubits = len(H.wires)
    num_param_sets = 2 ** num_qubits -1
    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))

    dev=qml.device('default.qubit',wires=num_qubits)

    cost_fn_gs=qml.ExpvalCost(variational_ansatz,H,dev)

    max_iterations= 300
    conv_tol=1e-7

    # Ground state

    opt = qml.AdamOptimizer(stepsize=0.4)

    for n in range(max_iterations):
        params, prev_cost = opt.step_and_cost(cost_fn_gs,params)
        cost = cost_fn_gs(params)
        conv = abs(cost - prev_cost)
        if n % 20 == 0:
            #print('Ground State: Iteration = {:},  Energy = {:.9f} , Convergence = {:.9f}'.format(n, cost,conv))
            energies[0]=cost
        if conv>=1e-3:
            opt.update_stepsize(0.4)
        elif 1e-5<conv<1e-3:
            opt.update_stepsize(0.2)
        #elif 1e-7<=conv<1e-5:
        #    opt.update_stepsize(0.1)
        #elif 1e-8<=conv<1e-7:
        #    opt.update_stepsize(0.05)
        elif conv_tol<conv<1e-5:
            opt.update_stepsize(0.1)
        elif conv <= conv_tol or n==max_iterations-1:
            energies[0]=cost
            #print('Ground State: Iteration = {:},  Energy = {:.9f} , Convergence = {:.9f}'.format(n, cost,conv))
            break
        else:
            continue

    ground_state=dev.state
    #First excited state

    #params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))

    opt=qml.AdamOptimizer(stepsize=0.4)
    max_iterations= 1000
    conv_tol=1e-9

    def costs_fes(params):
        energy_cost = qml.ExpvalCost(variational_ansatz, H, dev)(params)
        state = dev.state
        gs_cost = 3*abs(sum( a * np.conj(b)
                      for a, b in zip(state, ground_state))) ** 2
        return energy_cost, gs_cost
    #cost_fn_fes = lambda params: sum(costs_fes(params))


    for n in range(max_iterations):
        params, prev_cost = opt.step_and_cost(lambda params: sum(costs_fes(params)), params)
        cost = sum(costs_fes(params))
        conv = abs(cost - prev_cost)
        if n % 20 == 0:
            #print('first Excited State: Iteration = {:},  Energy = {:.9f} , Convergence = {:.9f}'.format(n, cost,conv))
            energies[1] = cost
        if conv>=1e-3:
            opt.update_stepsize(0.4)
        elif 1e-5<conv<1e-3:
            opt.update_stepsize(0.1)
        elif 1e-7<=conv<1e-5:
            opt.update_stepsize(0.05)
        elif conv_tol<conv<1e-7:
            opt.update_stepsize(0.01)
        elif conv <= conv_tol or n==max_iterations-1:
            energies[1] = cost
            #print('first Excited State: Iteration = {:},  Energy = {:.9f} , Convergence = {:.9f}'.format(n, cost,conv))
            break
        else:
            continue

    first_excited_state = dev.state
    #print(conv)
    #print(energies[1])
    #Second excited state

    #params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))

    opt = qml.AdamOptimizer(stepsize=0.4)
    def costs_ses(params):
        energy_cost = qml.ExpvalCost(variational_ansatz, H, dev)(params)
        state = dev.state
        gs_cost = abs(sum( a * np.conj(b)
                      for a, b  in zip(state, ground_state))) ** 2
        fes_cost = abs(sum( a * np.conj(b)
                      for a, b in zip(state, first_excited_state))) ** 2

        return energy_cost, 3*gs_cost, 9*fes_cost
    #cost_fn_ses =lambda params: sum(costs_fes(params,beta))

    #beta=np.array([1,3,9])
    #beta=beta/sum(beta)
    #beta=beta/np.sum(beta)
    for n in range(max_iterations):
        params, prev_cost = opt.step_and_cost(lambda params: sum(costs_ses(params)), params)
        cost = sum(costs_ses(params))
        conv = abs(cost - prev_cost)
        if n % 20 == 0:
            #print('2nd Excited State: Iteration = {:},  Energy = {:.9f} , Convergence = {:.9f}'.format(n, cost,conv))
            energies[2] = cost
        if conv>=1e-3:
            opt.update_stepsize(0.4)
        elif 1e-5<conv<1e-3:
            opt.update_stepsize(0.1)
        elif 1e-7<=conv<1e-5:
            opt.update_stepsize(0.03)
        elif 1e-8<=conv<1e-7:
            opt.update_stepsize(0.01)
        elif conv_tol<conv<1e-8:
            opt.update_stepsize(0.003)
        elif conv <= conv_tol or n>=max_iterations-1:
            energies[2] = cost
            #print('2nd Excited State: Iteration = {:},  Energy = {:.9f} , Convergence = {:.9f}'.format(n, cost,conv))

            break
        else:
            continue
            #print(beta[0:3])
    # 2.ans:        -1.31795925        ,-0.99412998        ,-0.32243601
    # Without beta: -1.3179565239425557,-0.9941294431983305,-0.3227860396664313
    # With beta:    -1.3179573760367984,-0.994129826376084 ,-0.32243613373223023
    #print(conv)
    #print(energies[2])
    # QHACK #
    #print(f'Total time : ',time.time()-clock)

    return ",".join([str(E) for E in energies])


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
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
