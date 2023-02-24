import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pdb
from matplotlib.pyplot import cm
import pickle

import seaborn as sns
import pandas as pd

with open('Results/SCGPP.pickle', 'rb') as pickle_file:
    temp = pickle.load(pickle_file)

scgpp_sensor = np.array([temp[i][:,-1] for i in range(len(temp))]).T

pga_nqp_path = 'Results/noise20000_epoch500_run500.npy'
pga_yahoo_path = 'Results/pga_yahoo_noise1000_epoch500_run100.npy'
scg_nqp_path = 'Results/scg_nqp_noise10000_epoch500_run100_fixed.npy'
scg_yahoo_path = 'Results/scg_yahoo_nosd_epoch300_run100.npy'
scgpp_nqp_path = 'Results/scgpp_nqp_noise10000_epoch751_run20.npy'
scgpp_yahoo_path = 'Results/scgpp_yahoo_nosd_epoch201_run100.npy'
boost_pga_nqp_path = 'Results/boost_pga_nqp_noise2000_run100.npy'
test = 'Results/pga_yahoo_nosd_epoch500_run100.npy'

data_path = test

result = scgpp_sensor#np.load(data_path)
print(result.shape)
# pdb.set_trace()
# if data_path == scgpp_yahoo_path:
#     data51 = np.load('Results/scgpp_yahoo_noise3_epoch51_run20_single.npy')
#     data60 = np.load('Results/scgpp_yahoo_noise3_epoch60_run20_single.npy')
#     data65 = np.load('Results/scgpp_yahoo_noise3_epoch65_run20_single.npy')
#     data75 = np.load('Results/scgpp_yahoo_noise3_epoch75_run20_single.npy')
#     data81 = np.load('Results/scgpp_yahoo_noise3_epoch81_run20_single.npy')
#     data90 = np.load('Results/scgpp_yahoo_noise3_epoch90_run20_single.npy')
#     data100 = np.load('Results/scgpp_yahoo_noise3_epoch100_run20_single.npy')
#
#     result = np.concatenate([result, data51, data60,  data75, data81],
#                             axis=1)
#     print(result[:, 0:1].shape, result[:, 2:].shape)
#     # result = np.concatenate([result[:, 0:1], result[:, 2:]], axis=1)
#     print(result.max(axis=0))

# plt.figure()
# for i in range(result.shape[0]):
#     plt.plot(result[i,:])
# plt.show()
def plot_confidence_diff(data, iters):
    percentiles = [5, 15, 25, 35, 45]
    expectation = np.mean(data, axis=0)
    plt.figure(figsize=(7.99, 6.2))

    def fit(x, const):
        return const / x**(1/2)#np.sqrt(x)
    plt.rcParams.update({'font.size': 23, 'legend.fontsize': 22})

    # popt_med, _ = curve_fit(fit, iters, np.abs(np.abs(np.percentile(data, 45, axis=0)-expectation) - expectation), method='lm')
    color = iter(cm.rainbow(np.linspace(0, 1, len(percentiles))))
    for percentile in percentiles:
        value = np.percentile(data, percentile, axis=0)
        popt, pcov = curve_fit(fit, iters[20:], np.abs(value-expectation)[20:], method='lm')
        print('constants,', popt, 'per', percentile)
        # popt[0] = 72

        c = next(color)
        plt.plot(iters, np.abs(value-expectation), c=c, label='percentile=%2.0f' % percentile)
        plt.plot(iters, fit(iters, *popt) / 1, '--', c=c)
        # print((data < bound).sum(axis=0))
        plt.legend(ncol=1)
        plt.ylabel('Distance Between Expectation And\nValues At Different Percentiles')
        plt.xlabel('Iteration (t)')
    # plt.savefig('Plots/pga_nqp_confidence', dpi=1000)
    plt.show()

def pga_variance_histogram(data):
    plt.figure(figsize=(7.99, 6.2))
    plt.rcParams.update({'font.size': 23, 'legend.fontsize': 21})
    variances = data.var(axis=1)
    plt.hist(variances, bins=20)
    plt.show()
# pga_variance_histogram(result)

def plot_box():
    pga_data = np.load('Results/noise20000_epoch500_run500.npy')[:, 10:]
    scg_data = np.load('Results/scg_nqp_noise5000_epoch100_run500_box.npy')
    scgpp_data = np.load('Results/scgpp_nqp_noise20000_epoch100_run500_box.npy')

    plt.figure(figsize=(7.99, 6.2))
    plt.rcParams.update({'font.size': 34, 'legend.fontsize': 21})
    df = pd.DataFrame(np.concatenate([pga_data.min(axis=1, keepdims=True) - pga_data.mean(),
                                      scg_data.reshape(scg_data.shape[0],1) - scg_data.mean(),
                                      scgpp_data.reshape(scgpp_data.shape[0],1) - scgpp_data.mean()],
                                     axis=1),
                 columns=['PGA', 'SCG', 'SCG++'])

    sns.boxplot(data=df[["PGA", "SCG", 'SCG++']], orient="v", width=0.2)
    # plt.axhline(y=pga_data.mean(), color='r', linestyle='--', label='expectation')
    plt.show()
# plot_box()
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
    plt.title('Sensor')
    plt.xlabel('Normalized Utility')
    plt.show()
# plot_max_min(result[:, 3:]/result.max())

if data_path == pga_yahoo_path or data_path == pga_nqp_path:
    step = np.tile(np.arange(500)+1, (len(result),1))
    result = np.cumsum(result, axis=1)/step
# result = result1
if data_path == boost_pga_nqp_path:
    step = np.tile(np.arange(200)+1, (len(result),1))
    result = np.cumsum(result, axis=1)/step

low, high = 50, 200

average = np.mean(result, axis=0)[low:high]
median = np.median(result, axis=0)[low:high]
percentile90 = np.percentile(result, 90, axis=0)[low:high]
minimum = np.min(result, axis=0)[low:high]
maximum = np.max(result, axis=0)[low:high]

iters = (np.arange(500)+1)[low:high]
#
if data_path == scg_yahoo_path:
    range_ = np.arange(300)[(low)*3:]
    iters = range_[range_ % 3 == 0]
if data_path == scgpp_nqp_path:
    range_ = np.arange(751)[(low)*10:high*10]
    iters = range_[range_ % 10 == 0]
if data_path == scgpp_yahoo_path:
    range_ = np.arange(201)[(low)*10:]
    iters = range_[range_ % 10 == 0]
# if data_path == scgpp_yahoo_path:
#     iters = np.array([2, 12, 22, 32, 42, 51, 60, 75, 100])

# iters = (np.arange(500)+1)[low:high]

method = 'lm'

# PGA: NQP, fit_offset=8050, conf_offset=8031, normalization=8050
#      Yahoo, offset=2.9, normalization=2.7
# plot_confidence_diff(result[:,low:high], iters)
offset = 2842
normalization = result.max()

def plot_variance(data, iters):
    plt.figure()
    plt.plot(iters, np.var(data, axis=0))
    plt.show()

# plot_variance(result[:,low:high], iters)

def fit_pga(x, c1, const):
    return c1 - const / np.sqrt(x)

def fit_pga_fix_offset(x, const):
    return offset - const / np.sqrt(x)

def fit_scg(x, c1, c2):
    return c1 - c2 / x**(1/3)

def fit_scg_fix_offset(x, const):
    return offset - const / x**(1/3)

def fit_scgpp(x, c1, c2):
    return c1 - c2 / x**(1/4)

def fit_scgpp_fix_offset(x, const):
    return offset - const / x**(1/4)

func = fit_pga_fix_offset
# popt_avg, pcov_avg = curve_fit(func, iters[1:], average[1:], method = method)
popt_min, pcov_min = curve_fit(func, iters[:], minimum[:], method = method)
popt_med, pcov_med = curve_fit(func, iters[:], median[:], method = method)
popt_90, pcov_90 = curve_fit(func, iters[:], percentile90[:], method = method)

# print('alpha', popt_avg)
print('alpha', popt_med)
print('beta', popt_min)
print('per', popt_90)

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
iters = iters[:]
def plot_fitted_curve():
    plt.figure(figsize=(7.99, 6.2))
    # plt.tight_layout()
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20, 'legend.columnspacing': 0.7})
    # plt.rc('figure.subplot', left=0.15, right=0.99, top=0.99, bottom=0.12)

    plt.plot(iters, func(iters, *popt_90)/normalization, 'r--',
             label='fit: c1=%5.2f, c2=%5.2f' % (offset/normalization, popt_90[0]/normalization))
    plt.plot(iters, func(iters, *popt_med)/normalization, 'c--',
             label='fit: c1=%5.2f, c2=%5.2f' % (offset/normalization, popt_med[0]/normalization))
    plt.plot(iters, func(iters, *popt_min)/normalization, 'b--',
             label='fit: c1=%5.2f, c2=%5.2f' % (offset/normalization, popt_min[0]/normalization))
    plt.plot(iters, percentile90[:] / normalization, 'r', label='90th percentile')
    plt.plot(iters, median[:]/normalization, 'c', label='median')
    plt.plot(iters, minimum[:]/normalization, 'b', label='min')

    # plt.plot(result.max(axis=0))
    # plt.plot(iters, average, label='average')

    if data_path == scg_yahoo_path:
        plt.legend(framealpha=0)
    else:
        plt.legend(framealpha=0)

    plt.legend(framealpha=0, ncol=1)
    if data_path == pga_yahoo_path or data_path == pga_nqp_path or data_path == boost_pga_nqp_path:
        plt.ylabel('Average Utility Up To t')
        plt.xlabel('Iteration (t)')
    else:
        plt.ylabel('Utility After Training T Iterations')
        plt.xlabel('Training Length (T)')
    # plt.plot(results.var(axis=0))
    # plt.savefig('Plots/pga_nqp_fit.png', dpi=500)
    # plt.ylim(0.9, 0.98)
    plt.show()
plot_fitted_curve()