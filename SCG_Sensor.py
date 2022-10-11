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
        #grad = [0] * self.num_sensors 
        grad = []
        iteration = 0

        for sensor in range(self.num_sensors):
            final_value += (time[sensor]*(1-(1-p)**x[sensor])*big_pi) 
            temp_grad = time[sensor]*((1-p)**(x[sensor]))*np.log((1-p)**(x[sensor]))*big_pi
            grad.append(temp_grad)
            #import pdb; pdb.set_trace()
            for i in range(iteration):
                temp_val = (-time[sensor]*(1-(1-p)**x[sensor])*(big_pi/(1-p)**x[i])*(1-p)**x[i]*np.log(1-p))
                grad[i] += temp_val 
            iteration += 1
            big_pi *= (1-p)**x[sensor] 
            #import pdb; pdb.set_trace()
            #gradient.append(grad) 
        import pdb; pdb.set_trace()
        return (self.t_inf - final_value), np.array(grad) + noise_scale 
    
    def setup_constraints(self):
        
        self.x = cp.Variable(shape=(self.num_sensors))
        self.p = cp.Parameter(shape=(self.num_sensors))
        #constraints = [cp.sum(self.x) == self.budget]
        constraints = [0 <= self.x , self.x <= self.budget]
        objective = cp.Maximize(self.x.T @ self.p)
        self.prob = cp.Problem(objective, constraints)

        
    def project(self, momentum):
        self.p.value = momentum 
        self.prob.solve(solver='ECOS')
        return self.x.value
    
    def stochastic_continuous_greedy_step(self, x, grad, p, momentum, epoch):
        new_momentum = (1-p)*momentum + p*grad        
        grad_projected= self.project(new_momentum)
        #@import pdb; pdb.set_trace()
        return x + 1/epoch * grad_projected, new_momentum
        
    def train(self, epoch, alpha, noise_scale=2000, step_coef=0.1):
        
        values = [] 
        for event in range(self.num_events):
            x = np.full((self.num_sensors), 0)
            momentum = np.zeros(self.num_sensors)            

            values_per_event = []
            value = 0
            for i in tqdm(range(epoch)):
                p = 4/((i+8)**(2/3))
                value, gradient = self.compute_val_grad(x, noise_scale, self.data[:,event], event)
                x, momentum = self.stochastic_continuous_greedy_step(x, gradient, p, momentum, epoch)
                values_per_event.append(value)      
            values.append(values_per_event)

        values = np.array(values).sum(axis=0) / self.num_events
        return values


 runs = 1
result = []
noise = 500
epoch = 200

budget = 4
data = mat['Z1'].toarray()[:10, :1000]
t_inf = float(np.max(data))
alpha = 1e-19
num_sensors, num_events = data.shape 
step_coef = 0.0001
pga = SCG_Sensor(data, num_sensors, num_events, budget, t_inf)
for _ in range(runs): 
    values = pga.train(epoch, alpha, noise_scale=noise, step_coef=step_coef)
    result.append(values)
result = np.array(result)

plt.plot(result[0])
plt.show()