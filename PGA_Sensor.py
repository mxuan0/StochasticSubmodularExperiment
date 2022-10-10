import cvxpy as cp
import numpy as np
from tqdm import tqdm
import scipy.io
import numpy as np
mat = scipy.io.loadmat('water_imp1000.mat')
import matplotlib.pyplot as plt

class PGA_Sensor:
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
        #constraints = [cp.sum(self.x) == self.budget]

        constraints = [0 <= self.x , self.x <= self.budget]
        objective = cp.Minimize(cp.sum_squares(self.x - self.p))
        self.prob = cp.Problem(objective, constraints)

        
    def project(self, x_t):
        self.p.value = x_t  
        self.prob.solve(solver='ECOS')
        return self.x.value

    def projected_gradient_ascent_step(self, x, grad, alpha): 
        x_t = x - alpha * grad
        #import pdb;pdb.set_trace()
        return self.project(x_t)

    def train(self, epoch, alpha, initialization=None, noise_scale=2000):
        values = []
        for event in range(self.num_events):
            x = np.full((self.num_sensors), 1e-5)
            if initialization is not None:
                x = initialization
            x = self.project(x)    

            values_per_event = []
            for i in tqdm(range(epoch)):
                value, gradient = self.compute_val_grad(x, noise_scale, self.data[:,event], event)
                x = self.projected_gradient_ascent_step(x, gradient, alpha)
                values_per_event.append(value)
            values.append(values_per_event)

        values = np.array(values).sum(axis=0) / self.num_events
        return values


runs = 5
result = []
noise = 100
epoch = 400

budget = 10
data = mat['Z1'].toarray()[:100, :100]
t_inf = float(np.max(data))
num_sensors, num_events = data.shape 
pga = PGA_Sensor(data, num_sensors, num_events, budget, t_inf)
for _ in range(runs): 
    values = pga.train(epoch, 1e-9, noise_scale=noise)
    result.append(values)
result = np.array(result)

plt.plot(result[0])
plt.show() 
