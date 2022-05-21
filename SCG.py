import cvxpy as cp
import numpy as np
import pdb
from YahooAdUtility import total_influence
from tqdm import tqdm

class SCG:
    def __init__(self, constraint_var, constraints) -> None:
        self.constraint_var = constraint_var
        self.constraints = constraints

    def project(self, momentum, constraints):
        x = self.constraint_var

        objective = cp.Maximize(cp.scalar_product(x, momentum))
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='OSQP')
        
        return x.value

    def compute_value_grad(self, x):
        return None

    def stochastic_continuous_greedy_step(self, x, grad, p, momentum, epoch):
        new_momentum = (1-p) * momentum + p * grad
        grad_projected = np.zeros_like(new_momentum)

        for i in range(new_momentum.shape[0]):
            grad_projected[i] = self.project(new_momentum[i], self.constraints)
        
        return x + 1/epoch * grad_projected, new_momentum

    def stochastic_continuous_greedy(self, epoch, weight_shape):
        x = np.zeros(shape=weight_shape) + 1e-5
        momentum = np.zeros(shape=weight_shape)

        values = []
        for e in tqdm(range(epoch)):
            pdb.set_trace()
            p = 4 / (e+8)**(2/3)
            value, grad = self.compute_value_grad(x)
            x, momentum = self.stochastic_continuous_greedy_step(x, grad, p, momentum, epoch)

            values.append(value)

        return values 

class SCG_Yahoo(SCG):
    def __init__(self, constraint_var, constraints, edge_prob, customer_to_phrase, advertiser_weights) -> None:
        super().__init__(constraint_var, constraints)

        self.edge_prob = edge_prob
        self.customer_to_phrase = customer_to_phrase
        self.advertiser_weights = advertiser_weights

    def compute_value_grad(self, x, noise_scale=0):
        noise = np.random.normal(scale=noise_scale, size=self.advertiser_weights.shape[0])

        influence, gradient = total_influence(self.advertiser_weights+noise, x, self.edge_prob, self.customer_to_phrase)

        return influence, gradient

advertiser_num = 10
phrase_num = 1001
weight_shape = (advertiser_num, phrase_num)
advertiser_weights = np.random.normal(loc=5, size=advertiser_num)
from YahooAdProcess import yahoo_ad_process
fn = 'data/YahooAdBiddingData/ydata-ysm-advertiser-bids-v1_0.txt'
customer_to_phrase, edge_weights, avp, phrase_price = yahoo_ad_process(fn)

x = cp.Variable(shape=(phrase_num))
constraints = [np.zeros_like(x) <= x, x <= np.array([0] + list(phrase_price)), cp.sum(x) <= avp]

pdb.set_trace()
scg = SCG_Yahoo(x, constraints, edge_weights, customer_to_phrase, advertiser_weights)
values = scg.stochastic_continuous_greedy(50, weight_shape)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(values)
plt.show()