import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pdb

import seaborn as sns
import pandas as pd

pga_nqp_path = 'Results/noise20000_epoch500_run500.npy'
pga_yahoo_path = 'Results/pga_yahoo_noise1000_epoch500_run100.npy'
scg_nqp_path = 'Results/scg_nqp_noise10000_epoch500_run100_fixed.npy'
scg_yahoo_path = 'Results/scg_yahoo_noise1000_epoch200_run100.npy'
scgpp_nqp_path = 'Results/scgpp_nqp_noise10000_epoch500_run100.npy'
test = 'Results/scg_yahoo_noise5_epoch201_run99.npy'

data_path = pga_yahoo_path
result = np.load(data_path)

def plot_confidence_diff(data, iters):
    percentiles = [5, 15, 25, 35, 45]
    minimum = np.mean(data, axis=0)
    plt.figure(figsize=(7.99, 6.2))

    def fit(x, c1, const):
        return c1 - const / np.sqrt(x)
    plt.rcParams.update({'font.size': 23, 'legend.fontsize': 22})
    for percentile in percentiles:
        value = np.percentile(data, percentile, axis=0)

        plt.plot(iters, np.abs(value-minimum), label='percentile=%2.2f' % percentile)
        # print((data < bound).sum(axis=0))
        plt.legend(ncol=1)
        plt.ylabel('Ratio of Trials Below Predicted\n Lower Bound')
        plt.xlabel('Iteration (t)')
    # plt.savefig('Plots/pga_nqp_confidence', dpi=1000)
    plt.show()

def pga_variance_histogram(data):
    plt.figure()
    variances = data.var(axis=1)
    plt.hist(variances, bins=20)
    plt.show()
# pga_variance_histogram(result)

def plot_box():
    pga_data = np.load(pga_nqp_path)[:, 10:]
    scg_data = np.load('Results/scg_nqp_noise20000_epoch5_run500_box.npy')

    plt.figure(figsize=(7.99, 6.2))
    plt.rcParams.update({'font.size': 23, 'legend.fontsize': 21})

    df = pd.DataFrame(np.concatenate([pga_data.min(axis=1, keepdims=True) - pga_data.mean(),
                                      scg_data.reshape(scg_data.shape[0],1)] - scg_data.mean(), axis=1),
                 columns=['PGA', 'SCG'])

    sns.boxplot(data=df[["PGA", "SCG"]], orient="v")
    # plt.axhline(y=pga_data.mean(), color='r', linestyle='--', label='expectation')
    plt.show()

plot_box()

def plot_max_min(data):
    plt.figure(figsize=(7.99, 6.2))
    plt.rcParams.update({'font.size': 23, 'legend.fontsize': 21})

    maxs = data.max(axis=1)
    mins = data.min(axis=1)

    expectation = data.mean()

    weights = np.ones_like(maxs) / len(maxs)

    plt.hist(maxs, bins=20, label='max', weights=weights)
    plt.hist(mins, bins=20, label='min', weights=weights)
    plt.axvline(x=expectation, color='r', linestyle='--', label='expectation')
    plt.legend()
    plt.title('NQP')
    plt.xlabel('Normalized Utility')
    plt.show()
# plot_max_min(result[:, 10:]/result.max())

if data_path == pga_yahoo_path or data_path == pga_nqp_path:
    step = np.tile(np.arange(500)+1, (len(result),1))
    result = np.cumsum(result, axis=1)/step
# result = result1

low, high = 10, 500

average = np.mean(result, axis=0)[low:high]
median = np.median(result, axis=0)[low:high]
minimum = np.min(result, axis=0)[low:high]
maximum = np.max(result, axis=0)[low:high]

iters = (np.arange(500)+1)[low:high]

if data_path == 'Results/scg_yahoo_noise5_epoch201_run99.npy':
    range_ = np.arange(201)[(low)*3:]
    iters = range_[range_ % 3 == 0]
method = 'lm'

# PGA: NQP, fit_offset=8050, conf_offset=8031, normalization=8050
#      Yahoo, offset=2.9, normalization=2.7
plot_confidence_diff(result[:,low:high], iters)

offset = 10542
normalization = 1

def plot_variance(data, iters):
    plt.figure()
    plt.plot(iters, np.var(data, axis=0))
    plt.show()

plot_variance(result[:,low:high], iters)

def fit_pga(x, c1, const):
    return c1 - const / np.sqrt(x)

def fit_pga_fix_offset(x, const):
    return offset - const / np.sqrt(x)

def fit_scg(x, c1, c2):
    return c1 - c2 / x**(1/3)

def fit_scg_fix_offset(x, const):
    return offset - const / x**(1/3)

def fit_scgpp(x, c1, c2):
    return c1 - c2 / x**(1/3)

def fit_scgpp_fix_offset(x, const):
    return offset - const / np.sqrt(x)

func = fit_pga
popt_avg, pcov_avg = curve_fit(func, iters, average, method = method)
popt_min, pcov_min = curve_fit(func, iters, minimum, method = method)
popt_med, pcov_med = curve_fit(func, iters, median, method = method)

# print('alpha', popt_avg)
print('alpha', popt_med)
print('beta', popt_min)

def frequency_by_confidence_pga(f, coef, data, iters, confidence:list):
    plt.figure(figsize=(7.99, 6.2))
    plt.rcParams.update({'font.size': 23, 'legend.columnspacing': 0.5, 'legend.fontsize': 20})
    for conf in confidence:
        coef_ = coef * np.sqrt(np.log(2/(1-conf))/np.log(4))
        # coef_[0] = 2.9000
        
        bound = f(iters, *coef_)
        freq = (data < bound).sum(axis=0)
        ratio = freq / data.shape[0]
        plt.plot(iters, ratio, label='conf=%2.2f'%conf)
        # print((data < bound).sum(axis=0))
        plt.legend(ncol=2)
        plt.ylabel('Ratio of Trials Below Predicted\n Lower Bound')
        plt.xlabel('Iteration (t)')
    # plt.savefig('Plots/pga_nqp_confidence', dpi=1000)
    plt.show()

confidence = [0.3, 0.5, 0.7, 0.8, 0.95, 0.99]
# frequency_by_confidence_pga(func, popt_med, result[:,low:high], iters, confidence)


percentiles = [5, 10, 20, 30, 40, 50]
def empirical_freq(data, iters, percentiles: list):
    plt.figure(figsize=(7.99, 6.2))
    #plt.rcParams.update({'font.size': 23, 'legend.columnspacing': 0.5, 'legend.fontsize': 20})
    for percentile in percentiles:
        value = np.percentile(data, percentile, axis=0)

        plt.plot(iters, value, label='percentile=%2.2f' % percentile)
        # print((data < bound).sum(axis=0))
        plt.legend(ncol=2)
        plt.ylabel('Ratio of Trials Below Predicted\n Lower Bound')
        plt.xlabel('Iteration (t)')
    # plt.savefig('Plots/pga_nqp_confidence', dpi=1000)
    plt.show()
# empirical_freq(result[:,low:high], iters, percentiles)

# plt.figure(figsize=(7.99, 6.2))
# plt.plot(iters, np.percentile(result[:,low:high], 50, axis=0) - np.percentile(result[:,low:high], 5, axis=0))
# plt.show()
def plot_fitted_curve():
    plt.figure(figsize=(7.99, 6.2))
    plt.rcParams.update({'font.size': 23, 'legend.fontsize': 21.5})
    # plt.rc('figure.subplot', left=0.15, right=0.99, top=0.99, bottom=0.12)

    # plt.plot(iters, func(iters, *popt_avg), '--',
    #          label='fit: alpha=%5.3f' % tuple(popt_avg))
    plt.plot(iters, func(iters, *popt_med)/normalization, 'r--',
             label='fit: c1=%5.2f, c2=%5.2f' % (offset/normalization, popt_med[0]/normalization))
    plt.plot(iters, func(iters, *popt_min)/normalization, 'b--',
             label='fit: c1=%5.2f, c2=%5.2f' % (offset/normalization, popt_min[0]/normalization))

    plt.plot(iters, median/normalization, 'r', label='median')
    plt.plot(iters, minimum/normalization, 'b', label='min')
    # plt.plot(result.max(axis=0))
    # plt.plot(iters, average, label='average')

    plt.legend()
    # plt.title('Utility vs Iteration')
    if data_path == pga_yahoo_path or data_path == pga_nqp_path:
        plt.ylabel('Average Utility Up To t')
        plt.xlabel('Iteration (t)')
    else:
        plt.ylabel('Utility After Training T Iterations')
        plt.xlabel('Training Length (T)')
    # plt.plot(results.var(axis=0))
    # plt.savefig('Plots/pga_nqp_fit', dpi=1000)
    plt.show()
plot_fitted_curve()