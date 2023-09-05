import numpy as np
from numpy.linalg import eig
import math
from numpy.linalg import matrix_power
import time


# C be a sufficiently large constant, set default = 0.29
def delete(data, episl, clean_e, C=0.29):
    start = time.time()
    delta = episl * np.sqrt(math.log(1 / episl))
    x = data[0]
    number = len(data[:, 0])
    n = len(data[:, 0])
    R = 4 * number
    constant_2 = 3
    dimension = len(data[0])
    # remove from T all points at distance more than R/2 from x
    delete_mem = np.ones(number)
    for i in range(number):
        dist = np.sqrt(np.sum((data[i] - x) ** 2))
        if dist <= R / 2:
            continue
        else:
            delete_mem[i] = 0
    delete_index = np.where(delete_mem == 0)[0].tolist()
    data = np.delete(data, delete_index, 0)
    # 3
    r = int(C * math.log(R / episl) * math.log(dimension))
    iteration = 0
    for i in range(r):
        number = len(data[:, 0])
        iteration += 1
        # a
        cov = np.cov(data.transpose())
        I = np.identity(dimension)
        # B = (number/n)*cov - (1 - C*(delta**2)/episl)*I
        # round it to the nearest integer

        M = matrix_power(((number / n) * cov - (1 - C * (delta ** 2) / episl) * I), int(math.log(dimension)))
        # b
        w = np.random.normal(0, 1, dimension)
        # c
        v = M @ (w.reshape(dimension, 1))
        # d
        m = np.mean(data, axis=0) @ v
        index = 0
        # e
        tau = np.zeros(number)
        f = abs(np.dot(data, v) - m) ** 2
        var = constant_2 * ((delta / episl) ** 2) * (np.linalg.norm(v) ** 2)
        mask = f > var;
        tau = mask * f
        #         for i in range(number):
        #             f = abs(data[i]@v - m)**2
        #             var = constant_2*((delta/episl)**2)* (np.linalg.norm(v)**2)
        #             if(f>var):
        #                 tau[i] = f
        #             else:
        #                 tau[i] = 0
        tau_sum = np.sum(tau)
        constraint = C * (delta ** 2 * n / episl) * (np.linalg.norm(v) ** 2)
        # try to work the same as the f
        while tau_sum > constraint:
            t = np.random.uniform(0, np.max(tau))
            delete_mem = np.ones(number)
            for i in range(number):
                if tau[i] <= t:
                    continue
                else:
                    delete_mem[i] = 0
            delete_index = np.where(delete_mem == 0)[0].tolist()
            data = np.delete(data, delete_index, 0)
            number = len(data[:, 0])
            tau = np.zeros(number)
            f = abs(np.dot(data, v) - m) ** 2
            var = constant_2 * ((delta / episl) ** 2) * (np.linalg.norm(v) ** 2)
            mask = f > var;
            tau = mask * f
            tau_sum = np.sum(tau)
        # if the error is small, stop the algorithm
        error = np.sum((0 - np.mean(data, axis=0)) ** 2)
        clean = clean_e
        # print(clean_e)
        if error - clean <= 0.0001:
            break
    mu = np.mean(data, axis=0)
    end = time.time()
    duration = end - start
    return mu, data, iteration, duration
