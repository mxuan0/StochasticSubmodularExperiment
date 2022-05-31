import cvxpy as cp
import numpy as np
import pdb
from YahooAdUtility import total_influence
from tqdm import tqdm

class PGA:
    def __init__(self, constraint_var, constraints) -> None:
        self.constraint_var = constraint_var
        self.constraints = constraints

    def project(self, x_infeasible, constraints):
        x = self.constraint_var

        objective = cp.Minimize(cp.sum_squares(x - x_infeasible))
        
        prob = cp.Problem(objective, constraints)
        prob.solve('ECOS')
        
        return x.value

    def compute_value_grad(self, x):
        return None

    def projected_gradient_ascent_step(self, x, grad, alpha):
        x_t = x + alpha * grad
        return self.project(x_t, self.constraints)

    def projected_gradient_ascent(self, epoch, weight_shape, alpha):
        x = np.random.randn(*weight_shape)
        x = self.project(x, self.constraints)    

        values = []
        for i in range(epoch):
            value, gradient = self.compute_value_grad(x)
            x = self.projected_gradient_ascent_step(x, gradient, alpha)

            values.append(value)
        return values 

class PGA_NQP(PGA):
    def __init__(self, constraint_var, constraints,
                 H, A, h, u_bar) -> None:
        super().__init__(constraint_var, constraints)
        self.H, self.A, self.h, self.u_bar = H, A, h, u_bar
        self.n = H.shape[1]
        self.m = A.shape[0]

    def compute_value_grad(self, x, noise_scale=10):
        noise = np.random.normal(scale=noise_scale, size=x.shape)
        
        value = 1/2*x @ self.H @ x.T + self.h.T @ x.T
        gradient = self.H@x.T + self.h
        
        return value[0][0], gradient + noise
'''
n = 100
m = 50
b = 1
u_bar = np.ones((1,n))
H = np.random.uniform(-100, 0, (n, n))
A = np.random.uniform(0, 1, (m, n))
h = -1 * H.T @ u_bar.T
train_iter = 50

x = cp.Variable(shape=(1,n))
constraints = [0 <= x, x <= u_bar, A @ x.T <= b]

pga = PGA_NQP(x, constraints, H, A, h, u_bar)
values = pga.projected_gradient_ascent(train_iter, (1,n), 1e-5)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(values)
plt.show()'''