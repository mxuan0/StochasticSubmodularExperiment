from cmath import nan
import numpy as np
from tqdm import tqdm
import pdb  
def influence_on_customer(x, customer_id, edge_prob, connected_phrases):
    '''
    x : vector, x[i] is the budget assignment on phrase i
    '''
    influence_ = 1
    for ph in connected_phrases:
        influence_ *= (1 - edge_prob[(ph,customer_id)]) ** x[ph]
    #pdb.set_trace()
    gradient = np.zeros_like(x)
    for ph in connected_phrases:
        gradient[ph] = - influence_ * x[ph] / (1 - edge_prob[(ph,customer_id)]) 

    return 1 - influence_, gradient

def influence_by_advertiser(x, edge_prob, customer_to_phrase):
    #assert((x >= np.zeros_like(x)).all() and (x[1:] <= budget_limit_phrase).all())
    #assert(x.sum() <= budget_limit_total)

    influence = 0
    gradient = np.zeros_like(x)
    for customer in customer_to_phrase:
        infl_, grad_ = influence_on_customer(x, customer, edge_prob, customer_to_phrase[customer])
        influence += infl_
        gradient += grad_
    return influence, gradient

def total_influence(weight_per_advertiser, budget_per_advertiser, edge_prob, customer_to_phrase):
    '''
    weight_per_advertiser: array of shape (number of advertisers,)
    budget_per_advertiser: array of shape (number of advertisers, number of phrases)
    '''
    influence = 0
    gradient = np.zeros_like(budget_per_advertiser)
    for i in range(len(weight_per_advertiser)):
        infl_, grad_ = influence_by_advertiser(budget_per_advertiser[i], edge_prob, customer_to_phrase)
        influence += weight_per_advertiser[i] * infl_
        gradient[i, :] = grad_

    return influence, gradient
    
'''
advertiser_num = 10
phrase_num = 1001
noise_scale = 0.1
intial_budget = np.zeros((advertiser_num, phrase_num))
weights = np.random.normal(scale=noise_scale, size=advertiser_num)

from YahooAdProcess import yahoo_ad_process
fn = 'data/YahooAdBiddingData/ydata-ysm-advertiser-bids-v1_0.txt'
customer_to_phrase, edge_weights, avp, phrase_price = yahoo_ad_process(fn)

print(total_influence(weights, intial_budget, edge_weights, customer_to_phrase))
'''