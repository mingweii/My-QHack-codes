#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #

    #import time
    #init_time=time.time()
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(params, x, depth=1):
        params = np.reshape(params,[3,3,depth])

        for layer in range(depth):
            qml.Rot(*x, wires=0)
            qml.Rot(*x, wires=1)
            qml.Rot(*x, wires=2)

            qml.Rot(*params[0,:,layer], wires=0)
            qml.Rot(*params[1,:,layer], wires=1)
            qml.Rot(*params[2,:,layer], wires=2)

        return [qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliX(wires=1)), qml.expval(qml.PauliX(wires=2))]

    def cost(params, x, y, depth):
        batch_loss = []
        label_dict = {
            1: [1, 0, 0],
            0: [0, 1, 0],
            -1: [0, 0, 1]
        }

        for i in range(len(x)):
            predict = circuit(params, x[i], depth=depth)
            label = label_dict[y[i]]

            loss = 0
            for l,p in zip(label,predict):
                loss += (l - p) ** 2
            loss = loss / len(label)

            batch_loss.append(loss)

        return sum(batch_loss)

    def batch_generator(X_batch, Y_batch, batch_size):
        for start_idx in range(0, X_batch.shape[0] - batch_size + 1, batch_size):
            yield X_batch[start_idx:start_idx+batch_size], Y_batch[start_idx:start_idx+batch_size]

    num_layers = 2
    eta = 0.1
    num_iterations = 10
    batch_size = 5

    opt = qml.AdamOptimizer(stepsize=eta)

    num_param_sets= 9 * num_layers
    params = np.random.uniform(low=0, high=np.pi,size=num_param_sets)

    for it in range(num_iterations):
        for Xbatch, ybatch in batch_generator(X_train, Y_train, batch_size=batch_size):
            params = opt.step(lambda v: cost(v, Xbatch, ybatch, num_layers), params)

    label_dict = {0:  1,
                  1:  0,
                  2: -1}
    for x in X_test:
        pred = circuit(params, x, depth=num_layers)
        label = label_dict[np.argmax(pred)]
        predictions.append(label)
    #print('Time: ',time.time()-init_time)
    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
