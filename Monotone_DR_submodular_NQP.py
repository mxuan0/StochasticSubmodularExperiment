import numpy as np
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

class monotoneDRsubmodularNQP:
    def __init__(self, H, A, h, u_bar) -> None:
        self.H, self.A, self.h, self.u_bar = H, A, h, u_bar
        self.n = H.shape[1]
        self.m = A.shape[0]

    #b = 10 constants = [(130000,40000), (900000,1900000), (-3500000,9000000)]
    def plot_accumulated_var(self, b=1, coefficients=[(160,50), (20000,15000), (-150000,460000)], 
                                labels=['A', 'B', 'C'], 
                                lr = [1e-3, 1e-4, 1e-5],
                                lr_scale=[3,4,5],
                                baseline=0.1, 
                                train_iter=50, expr_num=100):
        
        line = np.linspace(0.9, 50, num=50)
    
        Fs = []
        for alpha in lr:
            Fs.append(self.accumulated_values(b, alpha, train_iter, expr_num))
        
        fig, axs = plt.subplots(1, len(vars), figsize=(15, 3.5), dpi=500)
        fig.suptitle('b = 1', fontsize=17)
        for i in range(len(vars)):
            axs[i].plot(range(line.shape[0]), 
                        [coefficients[i][0]/line[j] + coefficients[i][1]/np.sqrt(line[j]) for j in range(line.shape[0])], 
                        '-.', label='%d/t+%d/sqrt(t)' % (coefficients[i][0], coefficients[i][1]))
            axs[i].plot(range(vars[i].shape[0]), vars[i], '-o', markersize= 3.5, label='ProjGrad lr = 1e-%d' % lr_scale[i])
            axs[i].plot(range(vars[i].shape[0]), [-10000 if vars[i][j] < baseline else baseline for j in range(len(vars[i]))], '--', c='k')
            #axs[i].legend(fontsize=12)
            axs[i].set_xlabel('Iteration (t)\n(%s)' % labels[i], fontsize=17)
            axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        #plt.plot(range(1, var1.shape[0]), var1[1:], '-o', label='non-accumulated')
        axs[0].set_ylabel('F-value Variance', fontsize=17)
        plt.show()
    
    def plot_var_nn(self, b, alpha, bound_constant, train_iter, expr_num):
        F_values = self.proj_grad_nn(b, alpha, train_iter, expr_num)
        line = np.linspace(0.9, train_iter, num=train_iter)
        var = np.var(F_values, axis=0)
        mean = np.mean(F_values, axis=0)

        fig, ax_left = plt.subplots(dpi=100)
        ax_right = ax_left.twinx()

        ax_left.plot(range(line.shape[0]), 
                    [bound_constant/np.sqrt(line[j]) for j in range(line.shape[0])], 
                    '-.', color='tab:blue')
        ax_left.plot(var, '-o', markersize= 3.5, color='darkorange')
        ax_right.plot(mean, '-o', markersize= 3.5, color='red')
        ax_left.set_ylabel('F-value Variance', fontsize=17)
        ax_left.set_xlabel('Iteration (t)\n(A)', fontsize=17)
        plt.show()
        
    def project(self, x_infeasible, b):
        x = cp.Variable(shape=(self.n,1))
        objective = cp.Minimize(cp.sum_squares(x - x_infeasible))
        constraints = [0 <= x, x <= self.u_bar, self.A @ x <= b]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return x.value

    def F(self, x):
        return 1/2*x.T @ self.H @ x + self.h.T @ x

    def F_gradient(self, x):
        return self.H@x + self.h

    def accumulated_values(self, b, alpha, train_iter, expr_num, noise_scale=10):
        Fs = []

        b_vec = np.ones([self.m,1])*b
        for _ in tqdm(range(expr_num)):
            l = []
            x_infeasible = np.random.randn(self.n,1)
            x_projected = self.project(x_infeasible, b_vec)      
            acc = 0
            
            for i in range(train_iter):
                gradient = self.F_gradient(x_projected) + np.random.normal(0, noise_scale, x_projected.shape)
                x_t = x_projected + alpha * gradient
                x_projected = self.project(x_t, b_vec)
                
                acc += self.F(x_projected).flatten()[0]
                l.append(acc/(i+1))
            Fs.append(l)
        
        return np.array(Fs).reshape((expr_num, train_iter))

    def proj_grad_nn(self, b, alpha, train_iter, expr_num, noise_scale=10, time_out=200):
        Fs = []

        b_vec = np.ones([self.m,1])*b
        for _ in tqdm(range(expr_num)):
            l = []
            counts = []
            x_infeasible = np.random.randn(self.n,1)
            x_projected = self.project(x_infeasible, b_vec)      

            for i in range(train_iter):
                count = 0
                
                while(count<time_out):
                    x_projected_ = x_projected
                    gradient = self.F_gradient(x_projected_) + np.random.normal(0, noise_scale, x_projected_.shape)
                    x_t = x_projected_ + alpha * gradient
                    x_projected_ = self.project(x_t, b) 
                    if self.F(x_projected_)[0][0] > self.F(x_projected)[0][0]:
                        break
                    count += 1
                
                x_projected = x_projected_ 
                l.append(self.F(x_projected).flatten()[0])
                counts.append(count)
                if count == time_out:
                    break
            Fs.append(l)
        return Fs       
'''
expr_num = 1
n = 100
m = 50
b = [1, 10]
u_bar = np.ones([n,1])
H = np.random.uniform(-100, 0, (n, n))
A = np.random.uniform(0, 1, (m, n))
h = -1 * H.T @ u_bar

train_iter = 50
alphas = [1e-3]
non_diag = ~np.eye(H.shape[0],dtype=bool)
H_ = np.where(non_diag == True, H , np.random.rand()*100)
nqp = monotoneDRsubmodularNQP(H,A,h,u_bar)
nqp.plot_var_nn(1, 1e-4, 5, 50, 1)'''
