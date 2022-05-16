import cvxpy as cp
import numpy as np
from YahooAdUtility import total_influence

class SCG:
    def __init__(self, constraints) -> None:
        self.constrants = constraints

    def project(self, momentum, constraints):
        x = cp.Variable(shape=momentum.shape)

        objective = cp.Maximize(cp.scalar_product(x, momentum))
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return x.value

    def compute_value_grad(self, x):
        return None

    def stochastic_continuous_greedy_step(self, x, grad, p, momentum, epoch):
        new_momentum = (1-p) * momentum + p * grad
        grad = np.zeros_like(new_momentum)

        for i in range(new_momentum.shape[0]):
            grad[i] = self.project(new_momentum[i], self.constraints)
        
        return x + 1/epoch * grad, new_momentum

    def stochastic_continuous_greedy(self, epoch, weight_shape=(10, 1001)):
        x = np.zeros(shape=weight_shape)
        momentum = np.zeros(shape=weight_shape)

        values = []
        for e in range(epoch):
            p = 4 / (e+8)**(2/3)
            value, grad = self.compute_value_grad(x)
            x, momentum = self.stochastic_continuous_greedy_step(x, grad, p, momentum, e)

            values.append(value)

        return values 

class SCG_Yahoo(SCG):
    def __init__(self, constraints, edge_prob, customer_to_phrase, advertiser_weights) -> None:
        super().__init__(constraints)

        self.edge_prob = edge_prob
        self.customer_to_phrase = customer_to_phrase
        self.advertiser_weights = advertiser_weights

    def compute_value_grad(self, x):
        influence, gradient = total_influence(self.advertiser_weights, x, self.edge_prob, self.customer_to_phrase)
    
        return influence, gradient
#constraints = [0 <= x, x <= phrase_price, cp.sum(x) <= total_price]