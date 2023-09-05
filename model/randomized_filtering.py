import numpy as np
from numpy.linalg import eig
import math
import time


def f(data, v, uT, episl):
    g = (np.dot((data - uT), v)) ** 2
    corr = int(episl * len(data[:, 0]))
    # get the index of the top corr element
    ind = np.argsort(g)[-corr:]
    f = np.zeros(len(data[:, 0]))
    for i in range(corr):
        f[ind[i]] = g[ind[i]]
    return f

# set C as a parameter
def delete(data, episl, C=0.000001):
    start = time.time()
    delta = episl * np.sqrt(math.log(1 / episl))
    lamda = delta ** 2 / episl
    cov = np.cov(data.transpose())
    uT = np.mean(data, axis=0)
    eigenvalue, eigenvector = eig(cov)
    v = eigenvector[:, 0]
    largest_eigenvalue = eigenvalue[0]
    iteration = 0
    while largest_eigenvalue >= (1 + C * lamda):
        number = len(data[:, 0])
        a = f(data, v, uT, episl)
        maximum = a.max()
        prob = a / maximum
        delete_mem = np.ones(number)
        for i in range(number):
            if prob[i] == 0:
                continue
            else:
                randprob = np.random.uniform(0, 1)
                if randprob <= prob[i]:
                    delete_mem[i] = 0
                else:
                    continue
        # delete the elements that are 0 in delete_mem
        delete_index = np.where(delete_mem == 0)[0].tolist()
        data = np.delete(data, delete_index, 0)
        cov = np.cov(data.transpose())
        uT = np.mean(data, axis=0)
        eigenvalue, eigenvector = eig(cov)
        v = eigenvector[:, 0]
        largest_eigenvalue = eigenvalue[0]
        iteration += 1
    mu = np.mean(data, axis=0)
    end = time.time()
    duration = end - start
    return mu, data, iteration, duration
