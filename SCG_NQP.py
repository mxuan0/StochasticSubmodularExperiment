from tabnanny import verbose
import cvxpy as cp
import numpy as np
import pdb
from tqdm import tqdm

class SCG_NQP:
    def __init__(self, H, A, h, u_bar, b) -> None:
        self.H, self.A, self.h, self.u_bar, self.b = H, A, h, u_bar, b
        self.n = H.shape[1]
        self.m = A.shape[0]

        self.setup_constraints()

    def sanity_check(self, x):
        if not (x < self.u_bar).all():
            pdb.set_trace()
        
        if not (self.A @ x < b).all():
            pdb.set_trace()

    def compute_value_grad(self, x, noise_scale=20000):
        noise = np.random.normal(scale=noise_scale, size=x.shape)
        
        value = 1/2*x.T @ self.H @ x + self.h.T @ x
        gradient = self.H@x + self.h
        #pdb.set_trace()
        return value[0][0], gradient + noise

    def setup_constraints(self):
        self.x = cp.Variable(shape=(self.n, 1))
        self.p = cp.Parameter(shape=(self.n, 1))
        
        constraints = [0 <= self.x, self.x <= self.u_bar, self.A @ self.x <= self.b]
        objective = cp.Maximize(self.x.T @ self.p)

        self.prob = cp.Problem(objective, constraints)

    def project(self, momentum):
        self.p.value = momentum
        self.prob.solve()
        
        return self.x.value

    def stochastic_continuous_greedy_step(self, x, grad, p, momentum, epoch):
        new_momentum = (1-p) * momentum + p * grad        
        grad_projected= self.project(new_momentum)
        
        #return x + 1e-2 * grad_projected, new_momentum
        return x + 1/epoch * grad_projected, new_momentum

    def train(self, epoch, c):
        x = np.zeros(shape=(self.n,1))
        momentum = np.zeros(shape=(self.n,1))

        values = []
        momentum_grad_diff = []
        for e in range(epoch):
            #pdb.set_trace()
            p = 4 / (e+8)**(2/3)
            #p = 1 / (e)**(2/3)
            value, grad = self.compute_value_grad(x)
            momentum_grad_diff.append(np.sum(np.square(grad-momentum)))
            x, momentum = self.stochastic_continuous_greedy_step(x, grad, p, momentum, (e+c)/1)
            #self.sanity_check(x)
            values.append(value)
            

        return x, values, momentum_grad_diff

n = 100
m = 50
b = 1

u_bar = np.ones((n,1))
H = np.random.uniform(-100, 0, (n, n))
A = np.random.uniform(0, 1, (m, n))
h = -1 * H.T @ u_bar

run = 50
train_iter = 20

for c in range(train_iter//2, train_iter, train_iter//10):
    results = []
    for _ in tqdm(range(run)):
        scg = SCG_NQP(H, A, h, u_bar, b)
        x, values, momentum_grad_diff = scg.train(train_iter, c)
        #pdb.set_trace()
        results.append(values[:])

    results = np.array(results)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(results.min(axis=0))
    plt.plot(results.max(axis=0))
    plt.plot(results.mean(axis=0)[:])
    #plt.plot(results.var(axis=0))
    #plt.show()
    plt.savefig('Plots/SCG_NQP_b1_c%d_noise20000.png' % c)