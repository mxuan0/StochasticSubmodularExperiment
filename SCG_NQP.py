from tabnanny import verbose
import cvxpy as cp
import matplotlib.pyplot as plt
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
        if not (np.abs((x - self.u_bar)[x > self.u_bar]) < 1e-7).all():
            pdb.set_trace()

        if not (np.abs((self.A @ x - self.b)[self.A @ x > self.b]) < 1e-3).all():
            pdb.set_trace()

    def compute_value_grad(self, x, noise_scale=2000):
        noise = np.random.normal(scale=noise_scale, size=x.shape)

        value = 1 / 2 * x.T @ self.H @ x + self.h.T @ x
        gradient = self.H @ x + self.h
        # pdb.set_trace()
        return value[0][0], gradient + noise

    def setup_constraints(self):
        self.x = cp.Variable(shape=(self.n, 1))
        self.p = cp.Parameter(shape=(self.n, 1))

        constraints = [0 <= self.x, self.x <= self.u_bar, self.A @ self.x <= self.b]
        objective = cp.Maximize(self.x.T @ self.p)

        self.prob = cp.Problem(objective, constraints)

    def project(self, momentum):
        self.p.value = momentum
        self.prob.solve('ECOS')

        return self.x.value

    def stochastic_continuous_greedy_step(self, x, grad, p, momentum, epoch):
        new_momentum = (1 - p) * momentum + p * grad
        grad_projected = self.project(new_momentum)

        return x + 1 / epoch * grad_projected, new_momentum

    def train(self, epoch, noise_scale=2000, alpha=None):
        x = np.zeros(shape=(self.n, 1))
        momentum = np.zeros(shape=(self.n, 1))

        # values = []
        momentum_grad_diff = []
        for e in range(epoch):
            # pdb.set_trace()
            p = 4 / (e + 8) ** (2 / 3)
            if alpha:
                p = 1 / (e+1)**alpha
            # p = 1 / (e)**(2/3)
            value, grad = self.compute_value_grad(x, noise_scale)
            momentum_grad_diff.append(np.sum(np.square(grad - momentum)))
            x, momentum = self.stochastic_continuous_greedy_step(x, grad, p, momentum,
                                                                 epoch)  # (e+1)/step_coef)#(e+c)/1)
            # values.append(value)
        self.sanity_check(x)

        return value

# n = 100
# m = 1
# b = 1
#
# u_bar = np.ones((n, 1))
# H = np.random.uniform(-50, 0, (n, n))
# H = H + H.T
# A = np.random.uniform(0, 1, (m, n))
# A = np.ones((1, n)) * b/n
# h = -1 * H.T @ u_bar
#
# run = 5
# epoch = 100
# noise_scale = 5000
# coef = 0.15
#
#
# scg = SCG_NQP(H, A, h, u_bar, b)
# results = []
# for _ in tqdm(range(run)):
#     iter_values = []
#     for train_iter in range(epoch):
#         try:
#             value = scg.train(train_iter, noise_scale=noise_scale)
#             iter_values.append(value)
#         except Exception as e:
#             print(e)
#             continue
#     results.append(iter_values)
#
# # for _ in tqdm(range(run)):
# #     try:
# #         value = scg.train(epoch, noise_scale=noise_scale)
# #         results.append(value)
# #     except Exception as e:
# #         value = scg.train(epoch, noise_scale=noise_scale)
# #         results.append(value)
#
# results = np.array(results)
# plt.figure()
# plt.plot(results.min(axis=0))
# plt.plot(results.max(axis=0))
# plt.show()
# np.save('Results/scg_nqp_noise%d_epoch%d_run%d_box.npy' % (noise_scale, epoch, run), results)
