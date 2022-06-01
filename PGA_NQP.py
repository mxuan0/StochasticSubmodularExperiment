import cvxpy as cp
import numpy as np
import pdb
from tqdm import tqdm

class PGA_NQP:
    def __init__(self, H, A, h, u_bar, b) -> None:
        self.H, self.A, self.h, self.u_bar, self.b = H, A, h, u_bar, b
        self.n = H.shape[1]
        self.m = A.shape[0]

        self.setup_constraints()

    def compute_value_grad(self, x, noise_scale=10):
        noise = np.random.normal(scale=noise_scale, size=x.shape)
        
        value = 1/2*x.T @ self.H @ x + self.h.T @ x
        gradient = self.H@x + self.h
        
        return value[0][0], gradient + noise

    def setup_constraints(self):
        self.x = cp.Variable(shape=(self.n, 1))
        self.p = cp.Parameter(shape=(self.n, 1))

        constraints = [0 <= self.x, self.x <= self.u_bar, self.A @ self.x <= self.b]
        objective = cp.Minimize(cp.sum_squares(self.x - self.p))

        self.prob = cp.Problem(objective, constraints)

    def project(self, x_t):
        self.p.value = x_t
        self.prob.solve(solver='ECOS')
        
        return self.x.value

    def projected_gradient_ascent_step(self, x, grad, alpha):
        x_t = x + alpha * grad
        return self.project(x_t)

    def train(self, epoch, alpha):
        x = np.random.randn(self.n, 1)
        x = self.project(x)    

        values = []
        for i in range(epoch):
            value, gradient = self.compute_value_grad(x)
            x = self.projected_gradient_ascent_step(x, gradient, alpha/(i+1))

            values.append(value)
        return values 

n = 100
m = 50
b = 1

u_bar = np.ones((n,1))
H = np.random.uniform(-100, 0, (n, n))
A = np.random.uniform(0, 1, (m, n))
h = -1 * H.T @ u_bar

alpha = 1e-4
train_iter = 100
run = 50

results = []
for _ in tqdm(range(run)):
    scg = PGA_NQP(H, A, h, u_bar, b)
    values = scg.train(train_iter, alpha)
    results.append(values[:])

results = np.array(results)

import matplotlib.pyplot as plt
plt.figure()
# plt.plot(results.min(axis=0))
# plt.plot(results.max(axis=0))
# plt.plot(results.mean(axis=0))
plt.plot(results.var(axis=0))
plt.show()