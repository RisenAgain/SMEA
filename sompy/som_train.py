import numpy as np
import math

import operator

def update_training_parameters(T, N,t,s,snot,tnot):
    """
    :param T: Total no. of generation
    :param N: Length of population
    :param t: current generation count
    :param s: current training data point
    :param snot: initial Sigma value
    :param tnot: initial Tau value
    :return:  updated Sigma and tau value
    """
    t1=t-1
    t2=t1*N
    t3=float(t2+s)
    t4=(t3/(T*N))
    t5=1-t4
    sigma= snot*t5
    tau = tnot *t5
    return sigma,tau


def closest_neuron(x, x_clusterno, neuron_weight, neuron_K, features):  #s_value, K_SOM[s], neuron_data, neuron_K
    """
    :param x_clusterno:  no. of clusters in solution x
    :param x: Current training solution
    :param features: no. of features in actual data
    :param neuron_weight: SOM neuron weights
    :param neuron_K: clusters corresponding to each neuron weight
    :return: find winning neuron index
    """
    distance={} # Dictionary to store distance
    new_y=[]
    # upd_y=[] # Stores only non zero values from y
    new_x=[] #

    for i in range(len(neuron_weight)):
        y=neuron_weight[i]
        for j in range(min((x_clusterno*features), (neuron_K[i]* features))):
                new_x.insert(j, x[j])
                new_y.insert(j, y[j])

        dist = np.linalg.norm(np.asarray(new_x) - np.asarray(new_y))
        distance[i] = dist
        new_y=[]
        new_x=[]
    sorted_x = sorted(distance.items(), key=operator.itemgetter(1))
    return sorted_x[0][0]


def build_lattice(population,cols):
    """
    
    :param population: Lattice weights
    :param cols: no. of columns in lattice
    :return: Lattice points for each neurons
    """
    lattice=[]
    for i in range (len(population)):
        row_no= i/cols
        col_no= i%cols
        index=[row_no,col_no]
        lattice.append(index)
    #print "lattice index values : ", lattice
    return lattice


def locate_neighbouring_neurons(winning_neuron,lattice,sigma):
    """
    
    :param winning_neuron: Index of winning neuron 
    :param lattice: SOM lattice coordinates
    :param sigma: Sigma value 
    :return: Dictionary of neighbouring neurons
    """
    index_of_winning_neuron=lattice[winning_neuron]
    distance={}
    neighbour_neuron={}
    k=0
    for i in range (len(lattice)):
        other_neuron=lattice[i]
        distance[i] = np.linalg.norm(np.asarray(index_of_winning_neuron) - np.asarray(other_neuron))
    for j in range((len(distance))):
        if distance[j]<sigma:
            neighbour_neuron[k]=distance[j]
        k+=1
    return neighbour_neuron


def update_weights(neuron_weight, neuron_K, neighbour_neuron, tau, training_data, K_training_data, feature):
    """
    :param neuron_weight: weight of neurons
    :param neuron_K: no. of clusters in each neuron weight
    :param neighbour_neuron: Dictionary of neighbouring neurons and their distance
    :param tau: updated learning rate (tau value obtained)
    :param training_data: current training solution
    :param K_training_data: No. of clusters for current training solution
    :param feature: no. of features in actual data
    :return: Updated weight of neurons
    """
    for i in range(len(neighbour_neuron)):
        k=0
        neighbour_neuron_weight=np.copy(neuron_weight[neighbour_neuron.keys()[i]])
        neighbour_neuron_K=np.copy(neuron_K[neighbour_neuron.keys()[i]])
        copy_neighbour_neuron_weight=list(neighbour_neuron_weight)

        a=K_training_data*feature
        b=neighbour_neuron_K*feature

        if a < b :  #check either training solution or neuron weight has minimum features.
            for ind in range (a,len(neighbour_neuron_weight)):
                neighbour_neuron_weight[ind] = 0

        if a>b:
            for ind in range(b, len(training_data)):
                training_data[ind] = 0

        if a==b:
            pass
        #print "length :", len(training_data), len(neighbour_neuron_weight)
        temp= tau*(math.exp(-(neighbour_neuron.values()[i])))*(np.asarray(training_data)-np.asarray(neighbour_neuron_weight))
        neuron_weight[neighbour_neuron.keys()[i]] = (np.asarray(copy_neighbour_neuron_weight) + np.asarray(temp))

    return neuron_weight