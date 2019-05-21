#==========================================
# Houston Methodist Research Institute
# May 21, 2019
# Supervisor: Vittorio Cristini
# Developer : Javier Ruiz Ramirez
#==========================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class AnalyticalSolution1D():

    def __init__(self, length = 1, components = 500):
        self.n_components = components
        self.L    = length
        self.time = 0

    def X(self, m,x):
        return np.sin(x * np.pi / self.L * (m + 0.5))

    def T(self, m,t):
        return np.exp(- t * (np.pi / self.L * (m + 0.5))**2 )

    def C(self, m):
        return -4/(np.pi * (2*m + 1))

    def V(self, m, x, t):
        return self.C(m) * self.T(m,t) * self.X(m,x)

    def Z(self, m, x, t):
        return np.exp(-t) * self.V(m,x,t)

    def U(self, x, t):
        s = x * 0 + 1
        for i in range(self.n_components):
            s += self.Z(i, x, t)
        return s

    def plot(self): 
        mesh_density = 1000
        x_mesh = np.linspace(0, 1, mesh_density)
        y = self.U(x_mesh, self.time)
        eps = 1e-2
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(x_mesh, y, 'b-')
        ax.set_xlim([0-eps,1+eps])
        ax.set_ylim([0-eps,1+eps])
        fig.savefig('sol.pdf', dpi=300)


