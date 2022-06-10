import cvxpy as cp
import numpy as np
import pdb
from YahooAdUtility import total_influence
from tqdm import tqdm
import math

class FW_NQP:
    def __init__(self, H, A, h, u_bar, b) -> None:
        self.H, self.A, self.h, self.u_bar, self.b = H, A, h, u_bar, b
        self.n = H.shape[1]
        self.m = A.shape[0]

        self.setup_constraints()

    def compute_value_grad(self, x, noise_scale=2000):
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

    def FW_step(self, x, grad, alpha, epoch):
        s = np.zeros_like(grad) 
        s= self.project(grad)
	
        return (1-alpha)*x + alpha*(s)        

    def train(self, epoch, noise_scale=2000, coef=0.1):
        x = np.zeros(shape=(self.n,1))
        momentum = np.zeros(shape=(self.n,1))

        values = []
        for e in tqdm(range(epoch)):
            alpha = coef/(e+1)
            #alpha = 0.1/(e+1)
            value, grad = self.compute_value_grad(x, noise_scale=noise_scale)
            x = self.FW_step(x, grad, alpha, epoch)

            values.append(value)

        return values 


n = 100
m = 50
b = 1

u_bar = np.ones((n,1))
H = np.random.uniform(-100, 0, (n, n))
A = np.random.uniform(0, 1, (m, n))
h = -1 * H.T @ u_bar

run = 20
train_iter = 500
noise_scale = 20000

iter_values = []
fw = FW_NQP(H, A, h, u_bar, b)

for _ in tqdm(range(run)):
  try:
    values = fw.train(train_iter, noise_scale=noise_scale, coef=.3)
    iter_values.append(values[:])
  except Exception as e:
    continue

results = np.array(iter_values)
import matplotlib.pyplot as plt
plt.figure()

plt.plot(results.min(axis=0))
plt.plot(results.max(axis=0))
plt.plot(results.mean(axis=0))
plt.show()





