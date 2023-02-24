import math

import matplotlib.pyplot as plt

from PGA_NQP import PGA_NQP
from SCG_NQP import SCG_NQP

import numpy as np
from tqdm import tqdm
from numpy.linalg import eig
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
import cvxpy as cp

# n = 2
# m = 1
# b = 1
#
# H = np.array([[-2, -1],[-1, -2]])
# A = np.array([[0.2, 0.1]])
n = 3
m = 1
b = 1

np.random.seed(1)
u_bar = np.ones((n, 1))
H = np.random.uniform(-1, 0, (n, n))
H = H + H.T
A = np.ones((1, n)) * b/n
h = -1 * H.T @ u_bar

alpha = 2/3
noise_scale = 0.1

# def objective(x):
#     return -np.square(x).sum()
#
# constraint1 = LinearConstraint(np.eye(n), lb=np.zeros(n), ub=u_bar.squeeze())
# constraint2 = LinearConstraint(A, lb=-np.inf, ub=b)
# res = minimize(objective, np.ones(n))
# print(res)

w,v=eig(H.T @ H)

M = 2 * noise_scale * np.sqrt(n)
D = np.sqrt(n)
L = np.sqrt(np.max(w))
C = (L+M)**2 + D**2 / 8
K = math.gamma(1/(1-alpha)) / (1-alpha)
sigma = n*noise_scale**2

opt = 6.5

run = 100
epoch = 500

print('M:', M, "| D:", D, "| L:", L, "| C:", C, "| K:", K, "| sigma:", sigma)

def run_PGA():
    alpha = 1e-4

    pga = PGA_NQP(H, A, h, u_bar, b)
    results = []
    for _ in tqdm(range(run)):
        while True:
            try:
                values = pga.train(epoch, alpha, noise_scale=noise_scale, var_step=True)
                break
            except Exception as e:
                continue
        results.append(values[:])

    results = np.array(results)
    np.save('Results/simpleNQP/pga.npy', results)

def run_SCG():
    scg = SCG_NQP(H, A, h, u_bar, b)
    results = []
    for _ in tqdm(range(50)):
        iter_values = []
        for i in range(1, 501, 10):
            try:
                value = scg.train(i, noise_scale=noise_scale, alpha=alpha)
                iter_values.append(value)
            except Exception as e:
                print(e)
                continue
        results.append(iter_values)

    results = np.array(results)
    print(results.max())
    np.save('Results/simpleNQP/scg500.npy', results)


def plot_PGA():
    result = np.load('Results/simpleNQP/pga.npy')

    step = np.tile(np.arange(epoch)+1, (len(result),1))
    result = np.cumsum(result, axis=1)/step

    low, high = 23, 500

    iters = np.arange(epoch)[low:high]
    minimum = np.min(result, axis=0)[low:high]
    median = np.median(result, axis=0)[low:high]
    percentile90 = np.percentile(result, 90, axis=0)[low:high]

    normalization = 1

    plt.figure()
    plt.plot(iters, percentile90[:] / normalization, 'r', label='90th percentile')
    plt.plot(iters, median[:]/normalization, 'c', label='median')
    plt.plot(iters, minimum[:]/normalization, 'b', label='min')

    plt.plot(iters, percentile90[:] / normalization, 'r', label='90th percentile')
    # plt.plot(iters, opt/2 - C/np.sqrt(iters) - D * M * np.sqrt(np.log(2)/(2*iters)), 'c--', label='median')
    plt.plot(iters, minimum[:]/normalization, 'b', label='min')

    plt.show()

def plot_SCG():
    result = np.load('Results/simpleNQP/scg500.npy')

    low, high = 1, 49

    iters = np.arange(1, 10000, 10)[low:high]
    minimum = np.min(result, axis=0)[low:high]
    median = np.median(result, axis=0)[low:high]
    percentile90 = np.percentile(result, 90, axis=0)[low:high]

    normalization = opt

    plt.figure()
    plt.rcParams.update({'font.size': 16, 'legend.fontsize': 16, 'legend.columnspacing': 0.7})
    # plt.plot(iters, percentile90[:] / normalization, 'r', label='90th percentile')
    # plt.plot(iters, median[:]/normalization, 'c', label='median')
    plt.plot(iters, minimum[:]/normalization, 'b', label='min')

    # plt.plot(iters, percentile90[:] / normalization, 'r', label='90th percentile')
    plt.plot(iters, (opt - (4*K+1)*L*D*D/2/iters - 2*D*K*sigma* np.sqrt(np.log(1/0.99)/iters))/normalization, 'r', label='prediction')
    plt.legend()
    # plt.plot(iters, minimum[:]/normalization, 'b', label='min')

    # plt.title('SCG')
    plt.ylabel('Normalized Utility')
    plt.xlabel('Iteration (t)')
    plt.savefig('Plots/simpleExample.png', dpi=500, bbox_inches = 'tight')
    plt.show()

# run_SCG()
plot_SCG()
# plot_PGA()
# run_PGA()