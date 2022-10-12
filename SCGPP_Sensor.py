import cvxpy as cp
import numpy as np
from tqdm import tqdm
import scipy.io
import numpy as np
mat = scipy.io.loadmat('water_imp1000.mat')
import matplotlib.pyplot as plt


class SCG_Sensor:
    def __init__(self, data, num_sensors, num_events, budget, t_inf) -> None:
        self.data = data
        self.num_sensors = num_sensors
        self.num_events = num_events
        self.budget = budget
        self.t_inf = t_inf
     
        self.setup_constraints()
        
    def compute_val_grad(self, x, noise_scale, time , event):
        time = np.sort(time)
        p = 0.0001 
        final_value = 0
        big_pi = 1
        #gradient = []
        grad = []
        
        for sensor in range(x.shape[0]):
            final_value += (time[sensor]*((1 -(1-p))**x[sensor])*big_pi)
            temp_grad = -time[sensor]*((1-p)**(x[sensor]))*np.log((1-p)**(x[sensor]))*big_pi
            grad.append(temp_grad)
            
            #grad = time[sensor]*x[sensor]*(1-(1-p))**(x[sensor] - 1)
            #gradient.append(temp)
            big_pi *= (1-p)**x[sensor] 
            #gradient.append(grad) 
        return (self.t_inf - final_value), -1*np.array(grad) + noise_scale
    
    def setup_constraints(self):
        
        self.x = cp.Variable(shape=(self.num_sensors))
        self.p = cp.Parameter(shape=(self.num_sensors))
        constraints = [cp.sum(self.x) == self.budget]
        objective = cp.Maximize(self.x.T @ self.p)
        self.prob = cp.Problem(objective, constraints)

        
    def project(self, momentum):
        self.p.value = momentum 
        self.prob.solve(solver='ECOS')
        return self.x.value
   
    def stochastic_continuous_greedy_step(self, x, grad, epoch):
        return x + 1 / epoch * self.project(grad)

        
    def train(self, epoch, noise_scale=2000, step_coef=0.1):
        
        values = [] 
        for event in range(self.num_events):
            x = np.full((self.num_sensors), 1e-5)
            momentum = np.zeros(self.num_sensors)            

            values_per_event = []
            for i in tqdm(range(epoch)):
                value, gradient = self.compute_val_grad(x, noise_scale, self.data[:,event], event)
                x = self.stochastic_continuous_greedy_step(x, gradient, epoch)
                values_per_event.append(value)
            
            values.append(values_per_event)

        values = np.array(values).sum(axis=0) / self.num_events
        return values


runs = 1
result = []
noise = 500
epoch = 200

budget = 10
data = mat['Z1'].toarray()[:129, :500]
t_inf = float(np.max(data))
num_sensors, num_events = data.shape 
step_coef = 0.15
pga = SCG_Sensor(data, num_sensors, num_events, budget, t_inf)
for _ in range(runs): 
    values = pga.train(epoch, noise_scale=noise, step_coef=step_coef)
    result.append(values)
result = np.array(result)

plt.plot(result[0])
plt.show() 
