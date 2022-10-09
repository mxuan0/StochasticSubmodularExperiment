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
        infl, gradient, _ = influence_by_advertiser(x, self.edge_prob, self.c_to_ph)
        #print(np.abs(gradient).max())
        return infl, gradient + noise

    def sanity_check(self, x):
        if not (np.abs((x - self.u_bar)[x > self.u_bar]) < 1e-5).all():
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

    def train(self, epoch, noise_scale=2000):
        values = []   
        for ad in range(self.num_advertiser):
            momentum = np.zeros(self.num_phrase)
            x = np.zeros(self.num_phrase)
            value = 0
            for e in range(epoch):
                p = 4 / (e+8)**(2/3)
                value, gradient = self.compute_value_grad(x, noise_scale)
                x, momentum = self.stochastic_continuous_greedy_step(x, gradient, p, momentum, epoch)

            values.append(value)
            self.sanity_check(x)
        return np.array(values).sum()

with open('data/YahooAdBiddingData/ADdata.pkl', 'rb') as inp:
    customer_to_phrase = pickle.load(inp)
    edge_weights = pickle.load(inp)
    avp = pickle.load(inp)
    phrase_price = pickle.load(inp)

result = []
num_advertiser, num_phrase = 1, 1000
noise = 1000
epoch = 20
runs = 1

scg = SCG_Yahoo(avp, num_advertiser, num_phrase, edge_weights, customer_to_phrase)
for _ in tqdm(range(runs)):
    iter_values = []
    for train_iter in range(epoch):
        try:
            value = scg.train(train_iter, noise_scale=noise)
            iter_values.append(value)
        except Exception as e:
            value = scg.train(train_iter, noise_scale=noise)
            iter_values.append(value)
    result.append(iter_values[:])
result = np.array(result)

print(result)
np.save('Results/scg_yahoo_noise%d_epoch%d_run%d.npy' % (noise, epoch, runs), result)



