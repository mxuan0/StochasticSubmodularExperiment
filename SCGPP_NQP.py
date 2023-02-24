import cvxpy as cp
import numpy as np
import pdb
from tqdm import tqdm

class SCGPP_NQP:
    def __init__(self, H, A, h, u_bar, b, batch_size0, batch_size) -> None:
        self.H, self.A, self.h, self.u_bar, self.b = H, A, h, u_bar, b
        self.batch_size0, self.batch_size = batch_size0, batch_size

        self.n = H.shape[1]
        self.m = A.shape[0]

        self.x_prev = np.zeros(shape=(self.n, 1))
        self.setup_constraints()

    def sanity_check(self, x):
        if not (np.abs((x - self.u_bar)[x > self.u_bar]) < 1e-3).all():
            pdb.set_trace()

        if not (np.abs((self.A @ x - self.b)[self.A @ x > self.b]) < 1e-3).all():
            pdb.set_trace()

    def compute_value_grad(self, x, step, noise_scale):
        if step == 0:
            noise = np.random.normal(scale=noise_scale, size=(self.n, self.batch_size0))\
                .mean(axis=1, keepdims=True)

            gradient = self.H @ x + self.h + noise
        else:
            noise = np.random.normal(scale=noise_scale, size=(self.n, self.n, self.batch_size)) \
                .mean(axis=2)
            gradient = self.grad_prev + (self.H + noise) @ (x - self.x_prev)

        value = 1 / 2 * x.T @ self.H @ x + self.h.T @ x
        return value[0][0], gradient

    def setup_constraints(self):
        self.v = cp.Variable(shape=(self.n, 1))
        self.p = cp.Parameter(shape=(self.n, 1))

        constraints = [0 <= self.v, self.v <= self.u_bar, self.A @ self.v <= self.b]
        objective = cp.Maximize(self.v.T @ self.p)

        self.prob = cp.Problem(objective, constraints)

    def project(self, momentum):
        self.p.value = momentum
        self.prob.solve('ECOS')

        return self.v.value

    def stochastic_continuous_greedy_step(self, x, grad, epoch):
        return x + 1 / epoch * self.project(grad)

    def train(self, epoch, noise_scale=2000):
        x = np.zeros(shape=(self.n, 1))

        value = 0
        for e in range(epoch):
            # pdb.set_trace()
            value, grad = self.compute_value_grad(x, e, noise_scale)
            x = self.stochastic_continuous_greedy_step(x, grad, epoch)#(e + 10) / step_coef)  # (e+c)/1)
            if e > 0:
                self.x_prev = x
            self.grad_prev = grad

        self.sanity_check(x)
        return x, value

n = 100
m = 50
b = 1

u_bar = np.ones((n, 1))
H = np.random.uniform(-100, 0, (n, n))
print(np.linalg.norm(H))
A = np.random.uniform(0, 1, (m, n))
h = -1 * H.T @ u_bar

runs = 500
noise = 100
epoch = 5

batch_size0 = epoch
batch_size = epoch

result = []
scg = SCGPP_NQP(H, A, h, u_bar, b, batch_size0, batch_size)

for _ in tqdm(range(runs)):
    x, value = scg.train(epoch, noise_scale=noise)
    result.append(value)

result = np.array(result)
np.save('Results/scgpp_nqp_noise%d_epoch%d_run%d_box.npy' % (noise, epoch, runs), result)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(result.min(axis=0))
# plt.plot(result.max(axis=0))
# plt.plot(result.mean(axis=0)[:])
# #plt.plot(results.var(axis=0))
# plt.show()

# run = 50
# train_iter = 20

# for c in range(5, 51, 5):
#     results = []
#     for _ in tqdm(range(run)):
#         scg = SCG_NQP(H, A, h, u_bar, b)
#         x, values, momentum_grad_diff = scg.train(train_iter, c)
#         #pdb.set_trace()
#         results.append(values[:])

#     results = np.array(results)

#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.plot(results.min(axis=0))
#     plt.plot(results.max(axis=0))
#     plt.plot(results.mean(axis=0)[:])
#     #plt.plot(results.var(axis=0))
#     #plt.show()
#     plt.savefig('Plots/SCG_NQP_b1_c%d_noise20000_const.png' % c)

# iter_values = []
# scg = SCG_NQP(H, A, h, u_bar, b)
# try:
#     for c in range(5, 200, 10):
#         last_values = []
#         for _ in tqdm(range(run)):
#             x, values, momentum_grad_diff = scg.train(c, c)
#             #pdb.set_trace()
#             last_values.append(values[-1])

#         iter_values.append((last_values, c))
# except cp.error.SolverError as e:
#     print(e)
# import matplotlib.pyplot as plt
# plt.figure()
# for last_values_, c_ in iter_values:
#     plt.scatter([c_ for _ in range(run)], last_values_, c='blue')


# plt.savefig('scg_dist_noise2000.png')
