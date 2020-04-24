import numpy as np

class OUNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, scaling = 1, x0=None):
        self.scaling = scaling
        self.theta = theta * scaling
        self.mu = mu
        self.sigma = sigma *scaling
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape, scale = self.scaling)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else self.mu #np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={}, scaling={})'.format(self.mu, self.sigma, self.scaling)

