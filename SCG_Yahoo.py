import cvxpy as cp
import numpy as np
import pdb, pickle
from tqdm import tqdm
from YahooAdUtility import total_influence, influence_by_advertiser

class SCG_Yahoo:
    def __init__(self, u_bar, num_advertiser, num_phrase, edge_prob, c_to_ph) -> None:
        self.u_bar = u_bar
        self.num_advertiser, self.num_phrase = num_advertiser, num_phrase
        self.edge_prob, self.c_to_ph = edge_prob, c_to_ph
        self.setup_constraints()

    def compute_value_grad(self, x, noise_scale=2000):
        noise = np.random.normal(scale=noise_scale, size=x.shape)
        infl, gradient = influence_by_advertiser(x, self.edge_prob, self.c_to_ph)
        #print(np.abs(gradient).max())
        return infl, gradient + noise

    def sanity_check(self, x):
        if not (x < self.u_bar).all():
            pdb.set_trace()


    def setup_constraints(self):
        self.x = cp.Variable(shape=(self.num_phrase))
        self.p = cp.Parameter(shape=(self.num_phrase))

        constraints = [0 <= self.x, self.x <= self.u_bar]
        objective = cp.Maximize(self.x.T @ self.p)

        self.prob = cp.Problem(objective, constraints)

    def project(self, x_t):
        self.p.value = x_t
        self.prob.solve(solver='ECOS')
        
        return self.x.value

    def stochastic_continuous_greedy_step(self, x, grad, p, momentum, epoch):
        new_momentum = (1-p) * momentum + p * grad        
        grad_projected= self.project(new_momentum)
        
        return x + 1/epoch * grad_projected, new_momentum

    def train(self, epoch, noise_scale=2000, step_coef=0.1):
        values = []   
        for ad in range(self.num_advertiser):
            momentum = np.zeros(self.num_phrase)
            x = np.zeros(self.num_phrase)+1e-5

            values_per_ad = []
            for e in tqdm(range(epoch)):
                p = 4 / (e+8)**(2/3)
                value, gradient = self.compute_value_grad(x, noise_scale)
                x, momentum = self.stochastic_continuous_greedy_step(x, gradient, p, momentum, (e+1)/step_coef)
                self.sanity_check(x)
                values_per_ad.append(value)

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
step_coef = 0.29
scg = SCG_Yahoo(avp, num_advertiser, num_phrase, edge_weights, customer_to_phrase)
for _ in range(1):
    values = scg.train(200,noise_scale=noise, step_coef=step_coef)
    result.append(values)
result = np.array(result)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(result.min(axis=0))
plt.plot(result.mean(axis=0))
plt.plot(result.max(axis=0))
plt.show()