from cmath import nan
import numpy as np
from tqdm import tqdm
import pdb
import time
def influence_on_customer(x, customer_id, edge_prob, connected_phrases, use_hessian):
    '''
    x : vector, x[i] is the budget assignment on phrase i
    '''
    influence_ = 1
    # time1 = time.time()
    log_prob = np.zeros((x.shape[0], 1))
    for phrase in connected_phrases:
        influence_ *= (1 - edge_prob[(phrase,customer_id)]) ** x[phrase]
        log_prob[phrase] = np.log(1 - edge_prob[(phrase, customer_id)])

        if np.isnan((1 - edge_prob[(phrase,customer_id)]) ** x[phrase]):
            pdb.set_trace()
    # print("time1", time.time() - time1)
    #pdb.set_trace()

    # hessian_time = time.time()
    if use_hessian:
        hessian = - influence_ * log_prob @ log_prob.T
    else:
        hessian = None
    # print("hessian time", time.time() - hessian_time)
    return 1 - influence_, - influence_ * log_prob.squeeze(), hessian #- influence_ * log_prob @ log_prob.T

def influence_by_advertiser(x, edge_prob, customer_to_phrase, use_hessian=False):
    #assert((x >= np.zeros_like(x)).all() and (x[1:] <= budget_limit_phrase).all())
    #assert(x.sum() <= budget_limit_total)

    influence = 0
    gradient = np.zeros_like(x)
    hessian = np.zeros((x.shape[0], x.shape[0]))

    for customer in customer_to_phrase:
        # time3 = time.time()
        infl_, grad_, hessian_ = influence_on_customer(x, customer, edge_prob, customer_to_phrase[customer], use_hessian)
        # print("time3", time.time() - time3)
        influence += infl_
        gradient += grad_
        if use_hessian:
            hessian += hessian_
    return influence, gradient, hessian

def total_influence(weight_per_advertiser, budget_per_advertiser, edge_prob, customer_to_phrase, use_hessian=False):
    '''
    weight_per_advertiser: array of shape (number of advertisers,)
    budget_per_advertiser: array of shape (number of advertisers, number of phrases)
    '''
    influence = 0
    gradient = np.zeros_like(budget_per_advertiser)
    for i in range(len(weight_per_advertiser)):
        infl_, grad_ = influence_by_advertiser(budget_per_advertiser[i], edge_prob, customer_to_phrase, use_hessian)
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