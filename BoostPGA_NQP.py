import cvxpy as cp
import numpy as np
import pdb
from tqdm import tqdm
from scipy import stats

class Z(stats.rv_continuous):
    def __init__(self, gamma, xtol=1e-14, seed=None):
        super().__init__(a=0, xtol=xtol, seed=seed)
        self.gamma = gamma
    def _pdf(self, x):
        return  self.gamma*np.exp(self.gamma*x) / (np.exp(self.gamma)-1)

class BoostPGA_NQP:
    def __init__(self, H, A, h, u_bar, b, gamma=1) -> None:
        self.H, self.A, self.h, self.u_bar, self.b = H, A, h, u_bar, b
        self.n = H.shape[1]
        self.m = A.shape[0]

        self.setup_constraints()
        self.zDist = Z(gamma)
        self.gamma = gamma

    def compute_value_grad(self, x, noise_scale=2000, clip_noise=False):
        noise = np.random.normal(scale=noise_scale, size=x.shape)
        if clip_noise:
            noise = noise.clip(-2 * noise_scale, 2 * noise_scale)
        z = self.zDist.rvs()
        value = 1 / 2 * x.T @ self.H @ x + self.h.T @ x
        gradient = z * self.H @ x + self.h
        self.grad_norm_acc += np.linalg.norm(gradient)
        return value[0][0], (1-np.exp(-self.gamma))/self.gamma * (gradient + noise)

    def setup_constraints(self):
        self.x = cp.Variable(shape=(self.n, 1))
        self.p = cp.Parameter(shape=(self.n, 1))

        constraints = [0 <= self.x, self.x <= self.u_bar, self.A @ self.x <= self.b]
        objective = cp.Minimize(cp.sum_squares(self.x - self.p))

        self.prob = cp.Problem(objective, constraints)

    def project(self, x_t):
        self.p.value = x_t
        self.prob.solve(solver='ECOS')
        # pdb.set_trace()
        return self.x.value

    def projected_gradient_ascent_step(self, x, grad, alpha):
        x_t = x + alpha * grad
        return self.project(x_t)

    def train(self, epoch, alpha, initialization=None, noise_scale=5000, var_step=False):
        x = np.random.randn(self.n, 1)
        # pdb.set_trace()
        if initialization is not None:
            x = initialization
        x = self.project(x)

        self.grad_norm_acc = 0
        values = []
        for i in range(epoch):
            if var_step:
                alpha = 0.002 / np.sqrt(i + 1)
            value, gradient = self.compute_value_grad(x, noise_scale)
            x = self.projected_gradient_ascent_step(x, gradient, alpha)

            values.append(value)

        print(self.grad_norm_acc / epoch)
        return values

n = 100
m = 50
b = 1

x = np.random.randn(n, 1)

u_bar = np.ones((n, 1))
H = np.random.uniform(-50, 0, (n, n))
H = H + H.T
A = np.random.uniform(0, 1, (m, n))
h = -1 * H.T @ u_bar

alpha = 1e-4
train_iter = 200
run = 500

#
# print(results)
# np.save('Results/pga_nqp_noise5000_run500_epoch500.npy', results)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(results.min(axis=0))
# plt.plot(results.max(axis=0))
# plt.plot(results.mean(axis=0))
# #plt.plot(results.var(axis=0))
# plt.show()

# iter_values = []
pga = BoostPGA_NQP(H, A, h, u_bar, b)
results = []
noise_scale = 2000

for _ in tqdm(range(run)):
    try:
        values = pga.train(train_iter, alpha, noise_scale=noise_scale, var_step=True)
        #pdb.set_trace()
        results.append(values)
    except Exception as e:
        print(e)
np.save('Results/boost_pga_nqp_noise%d_run%d.npy' % (noise_scale, run), np.array(results))
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
import matplotlib.pyplot as plt
plt.figure()
arr = np.load('Results/boost_pga_nqp_noise%d_run%d.npy' % (noise_scale, run))

rang = np.arange(train_iter)
plt.plot(rang, arr.min(axis=0))
plt.plot(rang,  np.median(arr, axis=0))
plt.plot(rang,  np.percentile(arr, 90, axis=0))
plt.show()
# for last_values_, c_ in iter_values:
#     plt.scatter([c_ for _ in range(run)], last_values_, c='blue')

# plt.savefig('pga_dist_noise2000.png')