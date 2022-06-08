from cProfile import label
import numpy as np
from scipy.optimize import curve_fit

result1 = np.load('/Users/jg/Downloads/ContinuousSubmodularMaximization/Results/noise2000_epoch500_run500.npy')
result2 = np.load('/Users/jg/Downloads/ContinuousSubmodularMaximization/Results/noise20000_epoch500_run500.npy')

step = np.tile(np.arange(500)+1, (len(result1),1))
result = np.cumsum(result2, axis=1)/step

average = np.mean(result, axis=0)[100:500]
minimum = np.min(result, axis=0)[100:500]
iters = (np.arange(500)+1)[100:500]

def func(x, const):
    return 8000 - const / np.sqrt(x)
popt_avg, pcov_avg = curve_fit(func, iters, average)
popt_min, pcov_min = curve_fit(func, iters, minimum)

print('alpha', popt_avg)
print('beta', popt_min)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(iters, func(iters, *popt_avg), '--',
         label='fit: alpha=%5.3f' % tuple(popt_avg))
plt.plot(iters, func(iters, *popt_min), '--',
         label='fit: beta=%5.3f' % tuple(popt_min))
plt.plot(iters, minimum, label='min')
# plt.plot(result.max(axis=0))
plt.plot(iters, average, label='average')
plt.legend()
#plt.plot(results.var(axis=0))
plt.show()
