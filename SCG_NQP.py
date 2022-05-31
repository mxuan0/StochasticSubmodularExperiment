import cvxpy as cp
import numpy as np
import pdb
from YahooAdUtility import total_influence
from tqdm import tqdm

class SCG_NQP:
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
        objective = cp.Maximize(self.x.T @ self.p)

        self.prob = cp.Problem(objective, constraints)

    def project(self, momentum):
        self.p.value = momentum
        self.prob.solve(solver='ECOS')
        
        return self.x.value

    def stochastic_continuous_greedy_step(self, x, grad, p, momentum, epoch):
        new_momentum = (1-p) * momentum + p * grad
        grad_projected = np.zeros_like(new_momentum)
        
        grad_projected= self.project(new_momentum)
        
        return x + 1/epoch * grad_projected, new_momentum

    def train(self, epoch):
        x = np.zeros(shape=(self.n,1))
        momentum = np.zeros(shape=(self.n,1))

        values = []
        for e in tqdm(range(epoch)):
            #pdb.set_trace()
            p = 4 / (e+8)**(2/3)
            value, grad = self.compute_value_grad(x)
            x, momentum = self.stochastic_continuous_greedy_step(x, grad, p, momentum, epoch)

            values.append(value)

        return values 

n = 100
m = 50
b = 1

u_bar = np.ones((n,1))
H = np.random.uniform(-100, 0, (n, n))
A = np.random.uniform(0, 1, (m, n))
h = -1 * H.T @ u_bar

train_iter = 10000

scg = SCG_NQP(H, A, h, u_bar, b)
values = scg.train(train_iter)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(values)
plt.show()
plt.savefig('Plots/SCG_NQP_b%d.png' % b)