import cvxpy as cp
import numpy as np
from tqdm import tqdm
import scipy.io
import numpy as np
mat = scipy.io.loadmat('water_imp1000.mat')
import matplotlib.pyplot as plt
import os; import time


class SCGPP_Sensor:
    def __init__(self, data, num_sensors, num_events, budget, t_inf, batch_size) -> None:
        self.data = data
        self.num_sensors = num_sensors
        self.num_events = num_events
        self.budget = budget
        self.t_inf = t_inf
        self.batch_size = batch_size

        self.x_prev = 0
        self.grad = 0
        self.setup_constraints()

    def compute_val_grad(self, x, step, noise_scale, time , event):
        time = np.sort(time)
        p = 0.1
        final_value = 0
        big_pi = 1
        #gradient = []
        #grad = [0] * self.num_sensors
        grad = []
        iteration = 0

        noise = np.random.normal(scale=noise_scale, size=x.shape)
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
        #import pdb; pdb.set_trace()


        grad = np.array(grad)
        hessian = grad.reshape(self.num_sensors,1) @ grad.reshape(1,self.num_sensors)
        if step == 0:
            noise = np.random.normal(scale=noise_scale, size=(self.num_sensors, self.batch_size))\
                .mean(axis=1)
            grad = grad + noise

        else:
            #import pdb; pdb.set_trace()
            grad = self.grad_prev + hessian @ (x - self.x_prev)

        return (self.t_inf - final_value), grad

    def setup_constraints(self):

        self.x = cp.Variable(shape=(self.num_sensors))
        self.p = cp.Parameter(shape=(self.num_sensors))
        constraints = [0 <= self.x , self.x <= self.budget]
        objective = cp.Maximize(self.x.T @ self.p)
        self.prob = cp.Problem(objective, constraints)


    def project(self, momentum):
        self.p.value = momentum
        self.prob.solve(solver='ECOS')
        return self.x.value

    def stochastic_continuous_greedy_step(self, x, grad, epoch):
        return x + 1/epoch*self.project(grad)


    def train(self, epoch, noise_scale=2000, step_coef=0.1):

        values = []
        for event in range(self.num_events):
            x = np.full((self.num_sensors), 1)
            momentum = np.zeros(self.num_sensors)

            values_per_event = []
            for i in tqdm(range(epoch)):
                value, gradient = self.compute_val_grad(x, i, noise_scale, self.data[:,event], event)
                x = self.stochastic_continuous_greedy_step(x, gradient, epoch)
                values_per_event.append(value)

                if i > 0:
                    self.x_prev = x
                self.grad_prev = gradient

            values.append(values_per_event)

        values = np.array(values).sum(axis=0) / self.num_events
        return values
