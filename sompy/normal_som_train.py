import numpy as np
import math

import operator


def update_training_parameters(T, N, t, s, snot, tnot):
    """
    :param T: Total no. of generation
    :param N: Length of population
    :param t: current generation count
    :param s: current training data point count
    :param snot: Sigma value
    :param tnot: Tau value
    :return:  updated Sigma and tau value
    """
    t1 = t - 1
    t2 = t1 * N
    t3 = float(t2 + s)
    t4 = (t3 / (T * N))
    t5 = 1 - t4
    sigma = snot * t5
    tau = tnot * t5
    return sigma, tau


def closest_neuron(x, population):
    """

    :param x: Current training data
    :param population: SOM neuron lattice
    :return: 
    """
    distance = {}  # Dictionary to store distance
    new_x = []  # Stores only non zero values from x i.e, current training data
    y = []  # values from SOM lattice at each iteration
    new_y = []
    upd_y = []  # Stores only non zero values from y
    upd_new_x = []  #
    new_x=np.copy(x)
    for i in range(len(population)):
        new_y = np.copy(population[i])


        print "Newx", new_x
        print "newy", new_y
        dist = np.linalg.norm(np.asarray(new_x) - np.asarray(new_y))
        distance[i] = dist
        new_y = []
        upd_y = []
        #new_x = list(new_x1)
        upd_new_x = []
    sorted_x = sorted(distance.items(), key=operator.itemgetter(1))
    print "Sorted x", sorted_x
    return sorted_x[0][0]


def build_lattice(population, cols):
    """

    :param population: Lattice weights
    :param cols: no. of columns in lattice
    :return: Lattice points for each neurons
    """
    lattice = []
    for i in range(len(population)):
        row_no = i / cols
        col_no = i % cols
        index = [row_no, col_no]
        lattice.append(index)
    print lattice
    return lattice


def locate_neighbouring_neurons(winning_neuron, lattice, sigma):
    """

    :param winning_neuron: Index of winning neuron 
    :param lattice: SOM lattice coordinates
    :param sigma: Sigma value 
    :return: Dictionary of neighbouring neurons
    """
    index_of_winning_neuron = lattice[winning_neuron]
    distance = {}
    neighbour_neuron = {}
    k = 0
    for i in range(len(lattice)):
        other_neuron = lattice[i]
        distance[i] = np.linalg.norm(np.asarray(index_of_winning_neuron) - np.asarray(other_neuron))
    for j in range((len(distance))):
        if distance[j] < sigma:
            neighbour_neuron[k] = distance[j]
        k += 1
    return neighbour_neuron


def update_weights(population, neighbour_neuron, tau, training_data):
    """

    :param population: updated SOM Population 
    :param neighbour_neuron: Dictionary of neighbouring neurons
    :param tau: tau value obtained
    :param training_data: Datapoint that is trained
    :return: Updated weight neuron lattice
    """
    updated_neighbour_neuron_weight = []
    updated_training_data = []
    train_nonzero_count1 = 0  # count value for training data non zero points
    train_zero_count1 = 0  # count value for training data zero points
    neighbour_nonZero_count1 = 0  # count value for neighbour data non zero points
    neighbour_zero_count1 = 0  # count value for neighbour data zero points

    for index in range(len(training_data)):
        if not training_data[index] == 0:
            train_nonzero_count1 += 1
        else:
            train_zero_count1 += 1

    for i in range(len(neighbour_neuron)):
        k = 0
        neighbour_neuron_weight = population[neighbour_neuron.keys()[i]]
        copy_neighbour_neuron_weight = list(neighbour_neuron_weight)
        for j in range(len(neighbour_neuron_weight)):
            if not neighbour_neuron_weight[j] == 0:
                neighbour_nonZero_count1 += 1
            else:
                neighbour_zero_count1 += 1

        if train_nonzero_count1 <= neighbour_nonZero_count1:
            for ind in range(train_nonzero_count1, len(neighbour_neuron_weight)):
                neighbour_neuron_weight[ind] = 0

        if train_nonzero_count1 > neighbour_nonZero_count1:
            for ind in range(neighbour_nonZero_count1, len(training_data)):
                training_data[ind] = 0
        temp = tau * (math.exp(-(neighbour_neuron.values()[i]))) * (
        np.asarray(training_data) - np.asarray(neighbour_neuron_weight))
        print (np.asarray(copy_neighbour_neuron_weight) + np.asarray(temp))
        print (np.asarray(training_data) - np.asarray(neighbour_neuron_weight))
        print tau * (math.exp(-(neighbour_neuron.values()[i])))
        population[neighbour_neuron.keys()[i]] = (np.asarray(copy_neighbour_neuron_weight) + np.asarray(temp))
        neighbour_neuron_weight = []
        neighbour_nonZero_count1 = 0  # Reset counters
        neighbour_zero_count1 = 0  # Reset counters
    return population