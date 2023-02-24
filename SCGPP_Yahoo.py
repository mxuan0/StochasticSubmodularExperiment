import pickle
import cvxpy as cp
import numpy as np
import pdb
from tqdm import tqdm
from multiprocessing import Process
from YahooAdUtility import influence_by_advertiser


class SCGPP_Yahoo:
    def __init__(self, u_bar, num_advertiser, num_phrase, edge_prob, c_to_ph) -> None:
        self.u_bar = u_bar
        self.num_advertiser, self.num_phrase = num_advertiser, num_phrase
        self.edge_prob, self.c_to_ph = edge_prob, c_to_ph

        self.x_prev = np.zeros(self.num_phrase)
        self.setup_constraints()

    def sanity_check(self, x):
        if not (x < self.u_bar).all():
            pdb.set_trace()

    def compute_value_grad(self, x, step, noise_scale):
        if step == 0:
            noise = np.random.normal(scale=noise_scale, size=(self.num_phrase, self.batch_size0)) \
                .mean(axis=1)

            influence, gradient, hessian = influence_by_advertiser(x, self.edge_prob, self.c_to_ph, use_hessian=True)
            gradient = gradient + noise
        else:
            influence, _, hessian = influence_by_advertiser(x, self.edge_prob, self.c_to_ph, use_hessian=True)
            gradient = self.grad_prev + \
                       (hessian + np.random.normal(scale=noise_scale,
                                                   size=(hessian.shape[0], hessian.shape[0], self.batch_size)) \
                        .mean(axis=-1)) @ (x - self.x_prev)

        return influence, gradient, hessian

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

    def train(self, epoch, noise_scale=2000):
        values = []
        self.batch_size0, self.batch_size = epoch, epoch

        for ad in range(self.num_advertiser):
            x = np.zeros(self.num_phrase)

            value = 0
            hessian_acc = np.zeros((self.num_phrase, self.num_phrase))
            for e in tqdm(range(epoch)):
                value, gradient, hessian = self.compute_value_grad(x, e, noise_scale)
                x = self.stochastic_continuous_greedy_step(x, gradient, epoch)  # (e+1)/step_coef)
                if e > 0:
                    self.x_prev = x
                self.grad_prev = gradient
                print(hessian)
                hessian_acc += hessian
            values.append(value)
            print('norm', np.linalg.norm(hessian_acc / epoch), 'shape', hessian_acc.shape)
        return np.array(values).sum()


def run_SCGPP_Yahoo(process_sequence_num):
    np.random.seed(process_sequence_num)

    with open('data/YahooAdBiddingData/ADdata.pkl', 'rb') as inp:
        customer_to_phrase = pickle.load(inp)
        edge_weights = pickle.load(inp)
        avp = pickle.load(inp)

    num_advertiser, num_phrase = 1, 1000

    runs = 1
    noise =3
    epoch = 20
    result = []

    scgpp = SCGPP_Yahoo(avp, num_advertiser, num_phrase, edge_weights, customer_to_phrase)

    for _ in tqdm(range(runs)):
        iter_values = []
        for train_iter in tqdm(range(epoch, epoch + 1)):
            if (train_iter) % 1 == 0:
                try:
                    value = scgpp.train(train_iter, noise_scale=noise)
                    iter_values.append(value)
                except Exception as e:
                    value = scgpp.train(train_iter, noise_scale=noise)
                    iter_values.append(value)
        result.append(iter_values[:])
    result = np.array(result)
    print(result)


if __name__ == '__main__':
    for num in range(1):
        Process(target=run_SCGPP_Yahoo, args=(num,)).start()

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
