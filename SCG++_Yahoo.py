import pickle
import cvxpy as cp
import numpy as np
import pdb
from tqdm import tqdm

from YahooAdUtility import influence_by_advertiser

class SCGPP_Yahoo:
    def __init__(self, u_bar, num_advertiser, num_phrase, edge_prob, c_to_ph,  batch_size0, batch_size) -> None:
        self.u_bar = u_bar
        self.num_advertiser, self.num_phrase = num_advertiser, num_phrase
        self.edge_prob, self.c_to_ph = edge_prob, c_to_ph

        self.batch_size0, self.batch_size = batch_size0, batch_size

        self.x_prev = np.zeros(self.num_phrase)
        self.setup_constraints()

    def sanity_check(self, x):
        if not (x < self.u_bar).all():
            pdb.set_trace()

    def compute_value_grad(self, x, step, noise_scale):
        if step == 0:
            noise = np.random.normal(scale=noise_scale, size=(self.num_phrase, self.batch_size0))\
                .mean(axis=1)

            influence, gradient, _ = influence_by_advertiser(x, self.edge_prob, self.c_to_ph, use_hessian=True)
            gradient = gradient + noise
        else:
            influence, _, hessian = influence_by_advertiser(x, self.edge_prob, self.c_to_ph, use_hessian=True)
            gradient = self.grad_prev + hessian @ (x - self.x_prev)

        return influence, gradient

    def setup_constraints(self):
        self.v = cp.Variable(shape=(self.num_phrase))
        self.p = cp.Parameter(shape=(self.num_phrase))

        constraints = [0 <= self.v, self.v <= self.u_bar]
        objective = cp.Maximize(self.v.T @ self.p)

        self.prob = cp.Problem(objective, constraints)

    def project(self, momentum):
        self.p.value = momentum
        self.prob.solve('ECOS')

        return self.v.value

    def stochastic_continuous_greedy_step(self, x, grad, epoch):
        return x + 1 / epoch * self.project(grad)

    def train(self, epoch, noise_scale=2000, step_coef=0.15):
        values = []
        for ad in range(self.num_advertiser):
            x = np.zeros(self.num_phrase)

            values_per_ad = []
            for e in tqdm(range(epoch)):
                value, gradient = self.compute_value_grad(x, e, noise_scale)
                x = self.stochastic_continuous_greedy_step(x, gradient, epoch) #(e+1)/step_coef)
                self.sanity_check(x)
                values_per_ad.append(value)

                if e > 0:
                    self.x_prev = x
                self.grad_prev = gradient

            values.append(values_per_ad)

        values = np.array(values).sum(axis=0)
        return values

with open('data/YahooAdBiddingData/ADdata.pkl', 'rb') as inp:
    customer_to_phrase = pickle.load(inp)
    edge_weights = pickle.load(inp)
    avp = pickle.load(inp)
    phrase_price = pickle.load(inp)

result = []
num_advertiser, num_phrase = 1, 1000
noise = 500
step_coef = 0.15
batch_size0 = 10
batch_size = 10

scg = SCGPP_Yahoo(avp, num_advertiser, num_phrase, edge_weights, customer_to_phrase, batch_size0, batch_size)
for _ in range(1):
    values = scg.train(200, noise_scale=noise, step_coef=step_coef)
    result.append(values)
result = np.array(result)


import matplotlib.pyplot as plt
plt.figure()
plt.plot(result.min(axis=0))
plt.plot(result.mean(axis=0))
plt.plot(result.max(axis=0))
plt.show()
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
