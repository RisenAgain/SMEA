from random import uniform
import numpy as np


def Repair_Solution( y, population, map_matrix, flag , x_K, x_count, b,a, features):
    '''

    :param map_matrix: Mapped matrix(neuron matrix after one to one mapping)
    :param y: Trial solution generated
    :param x: population
    :param x_count: main features of current solution y
    :return: Repaired solution of y
    '''
    yRep = []
    #t=0
    actual_length_y= len(y)
    # for kk in range(x_K):
    #     for fea in range(features):
    for k in range(x_count):
            if y[k] < a[k]:
                yRep.insert(k,a[k])
            elif y[k] > b[k]:
                yRep.insert(k,b[k])
            else:
                yRep.insert(k,y[k])

    # return_max.insert(k,xmax)
    # return_min.insert(k, xmin)
    #print "Yrev value ", k, "value", yRep[k]
    yRep.extend(0 for j in range(actual_length_y - x_count))  # Append extra zeros
    return yRep

def Select_Random(Q, mapped_matrix, population,i,flag, x):
    '''
    :param mapped_matrix: Codebook Matrix after one to one mapping
    :param population: Population
    :param Q: Mating Pool
    :param x: current solution
    :return: Random two parents from Q or population based on flag value
    '''
    print "Mating pool inside Random for solution  {} in population ".format(i),Q
    np.random.shuffle(Q)
    y1 = []
    y2 = []
    y3 = []
    if flag==0:

        indices_covered = []
        for i in range(len(Q)):
            rand_neuron1 = Q[i]
            if not np.array_equal(mapped_matrix[int(rand_neuron1)], x):
                # print "not equal for y1"
                indices_covered.append(i)
                for j in range(len(mapped_matrix[0])):
                    y1.insert(j, mapped_matrix[int(rand_neuron1)][j])
                break

        for i in range(len(Q)):
            rand_neuron2 = Q[i]
            if i not in indices_covered:
                # print "indices covered in list : ", rand_neuron2
                if not (np.array_equal(mapped_matrix[int(rand_neuron2)], x) and np.array_equal(mapped_matrix[int(rand_neuron2)], np.asarray(y1))):
                    # print "not equal for y2"
                    indices_covered.append(i)
                    for j in range(len(mapped_matrix[0])):
                        y2.insert(j, mapped_matrix[int(rand_neuron2)][j])
                    break

        # for i in range(len(Q)):
        #     rand_neuron3 = Q[i]
        #     if i not in indices_covered:
        #         # print "indices covered in list : ", rand_neuron2
        #         if not (np.array_equal(mapped_matrix[int(rand_neuron3)], x) and np.array_equal(mapped_matrix[int(rand_neuron3)], np.asarray(y1)) and np.array_equal(mapped_matrix[int(rand_neuron3)], np.asarray(y2))):
        #             # print "not equal for y2"
        #             indices_covered.append(i)
        #             for j in range(len(mapped_matrix[0])):
        #                 y3.insert(j, mapped_matrix[int(rand_neuron3)][j])
        #             break

        return y1, y2
    elif flag==1:

        #np.random.shuffle(Q)
        indices_covered = []
        for i in range(len(Q)):
            rand_neuron1 = Q[i]
            if not np.array_equal(population[int(rand_neuron1)], x):
                indices_covered.append(i)
                for j in range(len(population[0])):
                    y1.insert(j, population[int(rand_neuron1)][j])
                break
        for i in range(len(Q)):
            rand_neuron2 = Q[i]
            if i not in indices_covered and not (
                np.array_equal(population[int(rand_neuron2)], x) and np.array_equal(population[int(rand_neuron2)],
                                                                                    np.asarray(y1))):
                indices_covered.append(i)
                for j in range(len(population[0])):
                    y2.insert(j, population[int(rand_neuron2)][j])
                break

        # for i in range(len(Q)):
        #     rand_neuron3 = Q[i]
        #     if i not in indices_covered and not (np.array_equal(population[int(rand_neuron3)], x) and np.array_equal(population[int(rand_neuron3)],np.asarray(y1)) and np.array_equal(population[int(rand_neuron3)],np.asarray(y2))):
        #         indices_covered.append(i)
        #         for j in range(len(population[0])):
        #             y3.insert(j, population[int(rand_neuron3)][j])
        #         break
        return y1, y2


def Mutation(yRval,di,xmax,xmin):
    '''

    :param yRval: Value at ith index of Repaired solution
    :param x: Current population
    :param di: Mutated value
    :return:
    '''
    nsol=(float(yRval) + di * (xmax - xmin))
    #print "nlsolution part : ", nsol

    return nsol
