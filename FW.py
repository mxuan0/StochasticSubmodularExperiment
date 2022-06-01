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

    def FW_step(self, x, grad, alpha, epoch):
        s = np.zeros_like(grad) 
        s= self.project(grad)
	
        return (1-alpha)*x + alpha*(s)        

    def train(self, epoch):
        x = np.zeros(shape=(self.n,1))
        momentum = np.zeros(shape=(self.n,1))

        values = []
        for e in tqdm(range(epoch)):
            alpha = 0.01/(e+2)
            #alpha = 0.1/(e+1)
            value, grad = self.compute_value_grad(x)
            x = self.FW_step(x, grad, alpha, epoch)

            values.append(value)

        return values 