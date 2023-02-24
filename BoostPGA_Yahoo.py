import cvxpy as cp
import numpy as np
import pdb, pickle
from tqdm import tqdm
from YahooAdUtility import total_influence, influence_by_advertiser
from scipy import stats

class Z(stats.rv_continuous):
    def __init__(self, gamma, xtol=1e-14, seed=None):
        super().__init__(a=0, xtol=xtol, seed=seed)
        self.gamma = gamma
    def _pdf(self, x):
        return  self.gamma*np.exp(self.gamma*x) / (np.exp(self.gamma)-1)

class BoostPGA_Yahoo:
    def __init__(self, u_bar, num_advertiser, num_phrase, edge_prob, c_to_ph, gamma=1) -> None:
        self.u_bar = u_bar
        self.num_advertiser, self.num_phrase = num_advertiser, num_phrase
        self.edge_prob, self.c_to_ph = edge_prob, c_to_ph

        self.setup_constraints()

        self.zDist = Z(gamma)
        self.gamma = gamma

    def compute_value_grad(self, x, noise_scale=2000):
        z = self.zDist.rvs()
        noise = np.random.normal(scale=noise_scale, size=x.shape)
        infl, gradient, _ = influence_by_advertiser(z*x, self.edge_prob, self.c_to_ph)
        # print(np.abs(gradient).max())

        # pdb.set_trace()
        # gradient = np.ones_like(gradient) * 100
        return infl, (1-np.exp(-self.gamma))/self.gamma * (gradient + noise), gradient

    def setup_constraints(self):
        self.x = cp.Variable(shape=(self.num_phrase))
        self.p = cp.Parameter(shape=(self.num_phrase))

        constraints = [0 <= self.x, self.x <= self.u_bar]
        objective = cp.Minimize(cp.sum_squares(self.x - self.p))

        self.prob = cp.Problem(objective, constraints)

    def project(self, x_t):
        self.p.value = x_t
        self.prob.solve(solver='SCS')

        return self.x.value

    def projected_gradient_ascent_step(self, x, grad, alpha):
        x_t = x + alpha * grad
        return self.project(x_t)

    def train(self, epoch, alpha, initialization=None, noise_scale=2000):
        values = []
        for ad in range(self.num_advertiser):
            # x = np.random.randn(self.num_phrase)
            x = np.zeros(self.num_phrase) + 1e-5
            if initialization is not None:
                x = initialization
            # x = np.ones(self.num_phrase) * self.u_bar
            # pdb.set_trace()
            x = self.project(x)
            grad_acc = np.zeros_like(x)
            values_per_ad = []
            for i in tqdm(range(epoch)):
                value, gradient, clean_grad = self.compute_value_grad(x, noise_scale)
                x = self.projected_gradient_ascent_step(x, gradient, alpha/(1+i))
                grad_acc += clean_grad
                values_per_ad.append(value)

            values.append(values_per_ad)
            print('norm', np.linalg.norm(grad_acc / epoch), 'shape', grad_acc.shape)
        values = np.array(values).sum(axis=0)
        return values

    # open processed Yahoo data


with open('data/YahooAdBiddingData/ADdata.pkl', 'rb') as inp:
    customer_to_phrase = pickle.load(inp)
    edge_weights = pickle.load(inp)
    avp = pickle.load(inp)
    phrase_price = pickle.load(inp)

# define training parameters
runs = 100
result = []
num_advertiser, num_phrase = 1, 1000
noise = 100
epoch = 200

pga = BoostPGA_Yahoo(avp, num_advertiser, num_phrase, edge_weights, customer_to_phrase)
for _ in tqdm(range(runs)):
    values = pga.train(epoch, 1e-2, noise_scale=noise)
    result.append(values)
result = np.array(result)
print(result)
np.save('Results/boost_pga_yahoo_noise%d_epoch%d_run%d.npy' % (noise, epoch, runs), result)
