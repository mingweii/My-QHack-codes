#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)


    # QHACK #

    def parameter_shift_term_ij(params,i,j,s1,s2):
        shifted = params.copy()
        shifted[j] += s2
        shifted[i] += s1
        forward = circuit(shifted)  # forward evaluation
        shifted[i] -= 2*s1
        backward = circuit(shifted) # backward evaluation
        return 0.25 * (forward - backward)

    f0=circuit(weights)
    for i in range(len(weights)):
        shifted = weights.copy()
        shifted[i] += np.pi/2
        forward = circuit(shifted)  # forward evaluation:f(theta+s e_i)

        shifted[i] -= np.pi
        backward = circuit(shifted) # backward evaluation: f(theta-s e_i)
        gradient[i]= 0.5 * (forward - backward)
        hessian[i,i]=0.5 * (forward - f0*2+backward)

        for j in range(0,i):
            if j!=i:
                hessian[i,j]=parameter_shift_term_ij(weights,i,j,np.pi/2,np.pi/2)-parameter_shift_term_ij(weights,i,j,np.pi/2,-np.pi/2)
                hessian[j,i]=hessian[i,j].copy()
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
