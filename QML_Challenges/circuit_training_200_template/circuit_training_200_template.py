#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """
    #import matplotlib.pyplot as plt
    #nx.draw(graph,with_labels=True)
    #plt.show()

    max_ind_set = []

    # QHACK
    wires = range(NODES)
    # Defines the QAOA cost and mixer Hamiltonians
    cost_h, mixer_h = qml.qaoa.max_independent_set(graph,constrained=True)
    # Defines a layer of the QAOA ansatz from the cost and mixer Hamiltonians
    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)

    dev = qml.device('default.qubit', wires=wires)

    def circuit(params, **kwargs):
        #for w in wires:
        #    qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, N_LAYERS, params[0],params[1])

    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=wires)
    probs=np.zeros([N_LAYERS])
    #for i in range(N_LAYERS):
    #    probs[i]=probability_circuit(params[0,i],params[1,i])
    #print(probs)
    probs = probability_circuit(params[0,:], params[1,:])
    state_d=np.argmax(probs)
    def decimaltobinary(num):
        return bin(num).replace("0b","")
    q=str(decimaltobinary(state_d))
    #print(q)
    q_array=np.zeros([NODES])
    for i in range(len(q)):
        q_array[NODES-i-1]=q[len(q)-i-1]

    for i in range(NODES):
        if q_array[i]==1:
            max_ind_set.append(i)
    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
