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

    def compute_value_grad(self, x, noise_scale=2000):
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

    def train(self, epoch, alpha, initialization=None, noise_scale=2000):
        x = np.random.randn(self.n, 1)
        if initialization is not None:
            x = initialization
        x = self.project(x)    

        values = []
        for i in range(epoch):
            value, gradient = self.compute_value_grad(x, noise_scale)
            x = self.projected_gradient_ascent_step(x, gradient, alpha)

            values.append(value)
        return values 

# n = 100
# m = 50
# b = 1

# x = np.random.randn(n, 1)

# u_bar = np.ones((n,1))
# H = np.random.uniform(-50, 0, (n, n))
# H = H + H.T
# A = np.random.uniform(0, 1, (m, n))
# h = -1 * H.T @ u_bar

# alpha = 1e-4
# train_iter = 20
# run = 50

# results = []
# for _ in tqdm(range(run)):
#     scg = PGA_NQP(H, A, h, u_bar, b)
#     values = scg.train(train_iter, alpha, x)
#     results.append(values[:])

# results = np.array(results)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(results.min(axis=0))
# plt.plot(results.max(axis=0))
# plt.plot(results.mean(axis=0))
# #plt.plot(results.var(axis=0))
# plt.show()

# iter_values = []
# pga = PGA_NQP(H, A, h, u_bar, b)
# try:
#     for c in range(5, 200, 10):
#         last_values = []
#         for _ in tqdm(range(run)):
#             values = pga.train(c, alpha, initialization=x)
#             #pdb.set_trace()
#             last_values.append(sum(values[:])/c)

#         iter_values.append((last_values, c))
# except cp.error.SolverError as e:
#     print(e)
# import matplotlib.pyplot as plt
# plt.figure()
# for last_values_, c_ in iter_values:
#     plt.scatter([c_ for _ in range(run)], last_values_, c='blue')

# plt.savefig('pga_dist_noise2000.png')