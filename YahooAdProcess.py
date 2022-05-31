import collections
import pandas as pd
import numpy as np
import pdb
def yahoo_ad_process(filename, edge_type='by_total'):
    dtype = {
        'account_id':int,
        'phrase_id':int,
        'price':np.float32
    }

    df = pd.read_csv(filename, sep='\t', 
                     names=['timestamp', 'phrase_id', 'account_id', 'price', 'auto'],
                     dtype=dtype)
    
    pair_counts = df.groupby(['phrase_id','account_id'])['price'].count()
    phrase_price = df.groupby(['phrase_id'])['price'].mean()
    
    edge_weights = collections.defaultdict(np.float64)
    customer_to_phrase = collections.defaultdict(list)
    if edge_type=='by_customer':
        customer_count = df.groupby(['account_id'])['phrase_id'].count()
        for phrase_acct in list(pair_counts.keys()):
            edge_weights[phrase_acct] = pair_counts[phrase_acct] / customer_count[phrase_acct[1]]
            customer_to_phrase[phrase_acct[1]].append(phrase_acct[0])
    elif edge_type == 'by_total':
        total_edges = len(df)
        #total_edges = len(list(pair_counts.keys()))
        for phrase_acct in list(pair_counts.keys()):
            edge_weights[phrase_acct] = pair_counts[phrase_acct] / total_edges
            customer_to_phrase[phrase_acct[1]].append(phrase_acct[0])
            
    return customer_to_phrase, edge_weights, df['price'].max(), np.array(phrase_price)


'''fn = 'data/YahooAdBiddingData/ydata-ysm-advertiser-bids-v1_0.txt'
customer_to_phrase, edge_weights, avp, phrase_price = yahoo_ad_process(fn)
count = 0
for customer in customer_to_phrase:
    count += len(customer_to_phrase[customer])
print(count)'''