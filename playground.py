import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pdb

result1 = np.load('Results/noise20000_epoch500_run500_015.npy')
result2 = np.load('/Users/jg/Downloads/ContinuousSubmodularMaximization/Results/noise20000_epoch500_run500.npy')
result3 = np.load('Results/scg_noise20000_epoch500_run500.npy')

step = np.tile(np.arange(500)+1, (len(result1),1))
# result = np.cumsum(result2, axis=1)/step
result = result1

low, high = 50, 500

average = np.mean(result, axis=0)[low:high]
median = np.median(result, axis=0)[low:high]
minimum = np.min(result, axis=0)[low:high]
iters = (np.arange(500)+1)[low:high]

method = 'lm'

# def func(x, const):
#     return 8035 - const / np.sqrt(x)

def func(x, c1, c2):
    return 10600 - c2 / x**(1/3)

popt_avg, pcov_avg = curve_fit(func, iters, average, method = method)
popt_min, pcov_min = curve_fit(func, iters, minimum, method = method)
popt_med, pcov_med = curve_fit(func, iters, median, method = method)

# print('alpha', popt_avg)
print('alpha', popt_med)
print('beta', popt_min)

def frequency_by_confidence_pga(f, coef, data, iters, confidence:list):
    plt.figure()
    for conf in confidence:
        coef_ = coef * np.sqrt(np.log(2/(1-conf))/np.log(4))
        
        bound = f(iters, *coef_)
        freq = (data < bound).sum(axis=0)
        ratio = freq / data.shape[0]
        plt.plot(iters, ratio, label='conf=%2.2f'%conf)
        # print((data < bound).sum(axis=0))
        plt.legend()
        plt.ylabel('ratio of trials below predicted lower bound')
        plt.xlabel('Iteration (t)')
    plt.show()

confidence = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
# frequency_by_confidence_pga(func, popt_med, result[:,low:high], iters, confidence)

plt.figure()
# plt.plot(iters, func(iters, *popt_avg), '--',
#          label='fit: alpha=%5.3f' % tuple(popt_avg))
plt.plot(iters, func(iters, *popt_med), 'r--',
         label='fit: offset=%5.3f, alpha=%5.3f' % tuple(popt_med))
plt.plot(iters, func(iters, *popt_min), 'b--',
         label='fit: offset=%5.3f, beta=%5.3f' % tuple(popt_min))

plt.plot(iters, median, 'r', label='median')
plt.plot(iters, minimum, 'b', label='min')
# plt.plot(result.max(axis=0))
# plt.plot(iters, average, label='average')
plt.legend()
plt.title('Utility vs Iteration')
plt.ylabel('Utility at t')
plt.xlabel('Iteration (t)')
#plt.plot(results.var(axis=0))
plt.show()