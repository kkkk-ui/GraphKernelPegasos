import numpy as np
import GraphKernelFunc as kf


def pegasos(graphs, classes, iter, lamda):
    alpha = np.zeros(len(graphs))
    
    for t in range(iter):
        i_t = np.random.randint(len(graphs))

