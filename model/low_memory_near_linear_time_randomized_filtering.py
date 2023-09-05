import numpy as np
from numpy.linalg import eig
import math
from tqdm import tqdm
import random

episl = 0.1
# mean and standard deviation for the clean points
mu_c, sigma_c = 0, 1
# mean and standard deviation for the outliers
mu_o, sigma_o = 2, 0.5


# C = 100 for the case of mu_o = 2, sigma_o = 0.5(only 1 group of outlier)

def sample_point(episl, filters, dimension, k):
    pass_check = False
    # k = int(dimension/(np.log(1/episl)))
    while not pass_check:
        from_clean = 0
        prob = np.random.uniform(0, 1)
        # the point is from the clean dataset
        if prob > episl:
            from_clean = 1
            point = np.random.normal(mu_c, sigma_c, size=(dimension))
        # the point is not from the clean dataset
        else:
            #             # this is for only 1 group of outliers
            #             point = np.random.normal(mu_o, sigma_o, size = (dimension))
            # This is the case for muli outliers
            index_k = random.randint(0, k - 1)
            outlier_mean = np.zeros(dimension)
            outlier_mean[index_k] = 1
            outlier_mean = outlier_mean * np.sqrt(dimension)
            point = np.random.normal(outlier_mean, 0.0001, size=(dimension))
        # check whether it pass the filters
        if len(filters) == 0:
            pass_check = True
        else:
            pass_check = True
            for i in range(len(filters)):
                if i == 0:
                    if np.sqrt(np.sum((point - filters[0][0]) ** 2)) > filters[0][1]:
                        pass_check = False
                elif np.absolute(point @ filters[i][0] - filters[i][1]) ** 2 > filters[i][2]:
                    if np.absolute(point @ filters[i][0] - filters[i][1]) ** 2 > filters[i][3]:
                        pass_check = False

    return point, from_clean


def delete(episl, dimension, k, number, C=0.27):
    filters = []
    delta = episl * np.sqrt(math.log(1 / episl))
    orig_error_sum = 0
    for i in range(number):
        point, count = sample_point(episl, filters, dimension, k)
        orig_error_sum += (0 - np.mean(point)) ** 2
    R = 4 * number
    constant_2 = 3
    # remove from T all points at distance more than R/2 from x
    x, hhhgd = sample_point(episl, filters, dimension, k)
    filters.append([x, R / 2])
    # 3
    r = int(C * math.log(R / episl) * math.log(dimension))
    after_error_sum = 0
    run_iteration = 0
    # calculate the tau_sum for inliers only
    for run_iteration in range(r):
        # a, b, c,
        I = np.identity(dimension)
        w = np.random.normal(0, 1, dimension)
        v = w
        # create a mean vector
        sum_for_calculate_mean = np.zeros((dimension))
        count_clean = 0
        for i in range(number):
            point, count = sample_point(episl, filters, dimension, k)
            sum_for_calculate_mean += point
            count_clean += count
        if count_clean / number == 1.0:
            for i in range(number):
                point, count = sample_point(episl, filters, dimension, k)
                after_error_sum += (0 - np.mean(point)) ** 2
            break
        mu = sum_for_calculate_mean / number
        for i in range(int(math.log(dimension))):
            sum_t = np.zeros(dimension)
            for j in range(number):
                point, hhhgd = sample_point(episl, filters, dimension, k)
                sum_t += np.vdot(v, point - mu) * (point - mu) / (number - 1)
            v = sum_t - (1 - C * (delta ** 2) / episl) * I @ v
        # d
        # e
        m = mu @ v
        tau_sum = 0
        max_tau = 0
        var = constant_2 * ((delta / episl) ** 2) * (np.linalg.norm(v) ** 2)
        for i in range(number):
            point, count = sample_point(episl, filters, dimension, k)
            f = abs(point @ v.reshape(dimension, 1) - m) ** 2
            if (f > var):
                if f > max_tau:
                    max_tau = f
                tau_sum += f
            else:
                tau_sum += 0
        constraint = C * (delta ** 2 * number / episl) * (np.linalg.norm(v) ** 2)
        # try to work the same as the f
        upper_bound = max_tau
        while (tau_sum >= constraint):
            t = np.random.uniform(0, upper_bound)
            filters.append([v, m, var, t])
            tau_sum = 0
            count_clean = 0
            for i in range(number):
                point, count = sample_point(episl, filters, dimension, k)
                count_clean += count
                f = abs(point @ v.reshape(dimension, 1) - m) ** 2
                if (f > var):
                    tau_sum += f
                else:
                    tau_sum += 0
            if (count_clean / number * 100 == 100.0):
                break
            upper_bound = t
    return filters, run_iteration, float(orig_error_sum), float(after_error_sum)
