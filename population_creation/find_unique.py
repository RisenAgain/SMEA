import numpy as np
import random

def find_unique_elements(data, data_K, som_neurons):
    pop = []
    pop_K = []
    for j in range(len(data)):
        if data[j] not in pop:
            pop.append(data[j])
            pop_K.append(data_K[j])
        else:
            pass
    if len(pop) < som_neurons:
        left = som_neurons - len(pop)
        for j in range(left):
            idx = random.randint(0, len(data)- 1)
            pop.append(data[idx])
            pop_K.append(data_K[idx])
    return np.asarray(pop), pop_K