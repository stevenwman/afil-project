import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, dt, windup_limit=100.0):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.dt = dt
        self.integral = np.zeros_like(self.kp)
        self.prev_error = np.zeros_like(self.kp)
        self.windup_limit = windup_limit
    
    def update(self, error):
        p_term = self.kp * error

        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        i_term = self.ki * self.integral
        
        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative
        
        control = p_term + i_term + d_term
        
        self.prev_error = error.copy()
        
        return control, (p_term, i_term, d_term)