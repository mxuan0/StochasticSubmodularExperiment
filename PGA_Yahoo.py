import cvxpy as cp
import numpy as np
import pdb, pickle
from tqdm import tqdm
from YahooAdUtility import total_influence, influence_by_advertiser

class PGA_Yahoo:
    def __init__(self, u_bar, num_advertiser, num_phrase, edge_prob, c_to_ph) -> None:
        self.u_bar = u_bar
        self.num_advertiser, self.num_phrase = num_advertiser, num_phrase
        self.edge_prob, self.c_to_ph = edge_prob, c_to_ph
        self.setup_constraints()

    def compute_value_grad(self, x, noise_scale=2000):
        noise = np.random.normal(scale=noise_scale, size=x.shape)
        infl, gradient, _ = influence_by_advertiser(x, self.edge_prob, self.c_to_ph)
        #print(np.abs(gradient).max())

        # pdb.set_trace()
        # gradient = np.ones_like(gradient) * 100
        return infl, gradient+noise, gradient

    def setup_constraints(self):
        self.x = cp.Variable(shape=(self.num_phrase))
        self.p = cp.Parameter(shape=(self.num_phrase))

        constraints = [0 <= self.x, self.x <= self.u_bar]
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
        values = []
        for ad in range(self.num_advertiser):
            x = np.random.randn(self.num_phrase)

            if initialization is not None:
                x = initialization
            # x = np.ones(self.num_phrase) * self.u_bar
            pdb.set_trace()
            x = self.project(x)
            grad_acc = np.zeros_like(x)
            values_per_ad = []
            for i in tqdm(range(epoch)):
                value, gradient, clean_grad = self.compute_value_grad(x, noise_scale)
                x = self.projected_gradient_ascent_step(x, gradient, alpha)
                grad_acc += clean_grad
                values_per_ad.append(value)

            values.append(values_per_ad)
            print('norm', np.linalg.norm(grad_acc/epoch), 'shape', grad_acc.shape)
        values = np.array(values).sum(axis=0)
        return values 

# open processed Yahoo data
with open('data/YahooAdBiddingData/ADdata.pkl', 'rb') as inp:
    customer_to_phrase = pickle.load(inp)
    edge_weights = pickle.load(inp)
    avp = pickle.load(inp)
    phrase_price = pickle.load(inp)

# define training parameters
runs = 1
result = []
num_advertiser, num_phrase = 1, 1000
noise = 0.005
epoch = 200

pga = PGA_Yahoo(avp, num_advertiser, num_phrase, edge_weights, customer_to_phrase)
for _ in range(runs):
    values = pga.train(epoch, 1e2, noise_scale=noise)
    result.append(values)
result = np.array(result)
print(result)
np.save('Results/pga_yahoo_noise%d_epoch%d_run%d.npy' % (noise, epoch, runs), result)
