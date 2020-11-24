import math #importing necessary libraries#
import numpy as np
from numpy.linalg import norm

class Particle(object):

    def __init__ (self, dims):
        self.limits = 0.5
        self.omega = 1
        self.x = np.empty(dims)
        self.v = np.empty(dims)
        self.p = np.empty(dims)
        fp = 10**8
        for i in range (0, dims):
            self.x[i] = self.limits*2*(np.random.rand()-0.5)
            self.v[i] = 2*(np.random.rand()-0.5)
            self.p[i] = self.x[i]

    def updatePersonalBest(self, f):
        if (f<self.fp):
            self.fp=f
            for i in range (0, dims):
                self.p[i] = self.x[i]

    def getVelocity(self):
        mean_vel = 0
        for i in range (0, self.dims):
            mean_vel += self.v[i]*self.v[i]
        mean_vel = np.sqrt(mean_vel/self.dims)
        return mean_vel

    def updateVectors(self, g):
        for i in range (0, self.dims):
            self.v[i] = self.omega * self.v[i] + (self.alpha1 * np.random.rand() * (self.p[i]-self.x[i])+(self.alpha2*np.random.rand()*(g[i]-self.x[i])))
            self.x[i] += self.v[i]
            if np.abs(self.x[i])>10.0:
                self.x[i] = 10.0 * (np.random.rand() - 0.5)
