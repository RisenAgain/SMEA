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

    for k in range(x_count):
            if y[k] < a[k]:
                yRep.insert(k,a[k])
            elif y[k] > b[k]:
                yRep.insert(k,b[k])
            else:
                yRep.insert(k,y[k])

    yRep.extend(0 for j in range(actual_length_y - x_count))  # Append extra zeros
    return yRep

def Select_Random(Q, neuron_weights, population,i,flag, x):
    '''
    :param neuron_weights: Neuron weight matrix
    :param population: archive Population
    :param Q: Mating Pool
    :param x: current solution
    :return: Random two parents from Q or population based on flag value
    '''
    y1 = []
    y2 = []

    #print "flag: ", flag
    #print "current sol ttype : ", type(x)
    #np.random.shuffle(Q)
    #print "Mating pool inside Random pick solution for solution  {} in population ".format(i),Q
    indices_covered = []
    # if flag==0:
    np.random.shuffle(Q)
    for i in range(len(Q)):
        rand_neuron1 = Q[i]
        # print "current solution : ", x[:10], type(x)
        # print "neuron number : ", rand_neuron1, type(rand_neuron1)
        #print "mapped matrix : ", neuron_weights[int(rand_neuron1)][:10], type(neuron_weights[int(rand_neuron1)])
        if not np.array_equal(population[int(rand_neuron1)], x):
            #print "not equal for y1"
            indices_covered.append(i)
            for j in range(len(x)):
                y1.insert(j, population[int(rand_neuron1)][j])
            break


    for i in range(len(Q)):
        rand_neuron2 = Q[i]
        # print "current solution : ", x[:10], type(x)
        # print "neuron number : ", rand_neuron2, type(rand_neuron2)
        # print "mapped matrix : ", neuron_weights[int(rand_neuron2)][:10], type(neuron_weights[int(rand_neuron2)])
        if i not in indices_covered:
             #print "indices covered in list : ", rand_neuron2
             if not (np.array_equal(population[int(rand_neuron2)], x) or np.array_equal(population[int(rand_neuron2)], np.asarray(y1))):
                #print "not equal for y2"
                indices_covered.append(i)
                for j in range(len(x)):
                    y2.insert(j, population[int(rand_neuron2)][j])
                break
    if len(y1)==0:
        a=x * uniform(0, 1)
        return a.tolist(), y2
    elif len(y2)==0:
        b = x * uniform(0, 1)
        return y1, b.tolist()
    elif (len(y1) > 0 and len(y2) > 0):
        return y1, y2
    else:
        a = x * uniform(0, 1)
        b = x * uniform(0, 1)
        return a.tolist(), b.tolist()

def Mutation(yRval,di,xmax,xmin):
    '''

    :param yRval: Value at ith index of Repaired solution
    :param x: Current population
    :param di: Mutated value
    :return:
    '''
    nsol=(yRval + di * (xmax - xmin))
    #print "nlsolution part : ", nsol

    return nsol
