#==========================================
# Houston Methodist Research Institute
# May 21, 2019
# Supervisor: Vittorio Cristini
# Developer : Javier Ruiz Ramirez
#==========================================

import fenics as fe
import numpy as np
import mshr as mesher
#import sympy 
import os
import re
import cv2


import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
from scipy.integrate import simps as srule

from image_manipulation_module import ImageManipulation

class RightDirichletNeumann():

#==================================================================
    def __init__(self):

        self.label        = 'RDN'
        self.n_components = 1000
        self.L            = 1
        self.time         = 0
        self.fast_run     = False

        #self.set_fem_properties()
        self.mode         = 'exp'
        self.dimension    = 1
        self.polynomial_degree = 2
        self.mesh_density = 50
        self.dt           = 0.1
        self.x_left       = 0
        self.x_right      = self.L
        self.fps          = 3

        self.image_manipulation_obj = ImageManipulation()

        self.initial_time    = 0
        self.final_time      = 2.0
        self.current_time    = None
        self.counter         = 0
        self.boundary_points = None
        self.figure_format   = '.png'
        self.video_format    = '.webm'

        self.alpha        = 0
        self.beta         = 0
        self.lam          = 1.0
        self.kappa        = 0
        self.diffusion_coefficient = 1.0

        self.error_list   = []
        self.u_true       = None
        self.rhs_fun_str  = None
        self.boundary_fun = None
        self.boundary_conditions = None
        self.mesh         = None
        self.u            = None
        self.u_n          = None
        self.rhs_fun      = None
        self.ic_fun       = None
        self.bilinear_form= None
        self.rhs          = None
        self.dof_map      = None
        self.ordered_mesh = None
        self.function_space = None

        self.current_dir  = os.getcwd()
        self.postprocessing_dir = os.path.join(self.current_dir,\
                'postprocessing')
        self.fem_solution_storage_dir = os.path.join(self.current_dir,\
                'solution', self.label, str(self.dimension) + 'D')
        self.movie_dir = os.path.join(self.fem_solution_storage_dir, 'movie')

        self.u_experimental = None

        self.plot_true_solution= False
        self.plot_symmetric    = False

#==================================================================
    def set_data_dirs(self):

        if not os.path.exists(self.postprocessing_dir):
            os.makedirs(self.postprocessing_dir)

        if not os.path.exists(self.fem_solution_storage_dir):
            os.makedirs(self.fem_solution_storage_dir)

        if not os.path.exists(self.movie_dir):
            os.makedirs(self.movie_dir)

        txt   = 'solution.pvd'
        fname = os.path.join(self.fem_solution_storage_dir, txt)
        self.vtkfile = fe.File(fname)

#==================================================================
    def create_rhs_fun(self):

        self.rhs_fun = fe.Constant(1)

#==================================================================
    def create_boundary_conditions(self):

        if self.dimension == 2:
            self.boundary_fun = fe.Constant(1)
            def is_on_the_boundary(x, on_boundary):
                return on_boundary and 4.34 < x[0]

        tol = 1e-6
        if self.dimension == 1:

            self.boundary_fun =\
                    fe.Expression('x[0] < 1e-6 ? 1 : 0', degree=0)

            def is_on_the_boundary(x, on_boundary):
                return on_boundary and False

        self.boundary_conditions = fe.DirichletBC(\
                self.function_space, self.boundary_fun,\
                is_on_the_boundary)

#==================================================================
    def set_initial_conditions(self):

        self.current_time = self.initial_time 

        if self.dimension == 2:
            #self.u_n = fe.project(self.ic_fun, self.function_space)
            pass

        if self.dimension == 1:
            sigmoidal = fe.Expression('1/( 1 + exp(30*(x[0]-0.5)))', degree=2)
            self.u_n = fe.interpolate(sigmoidal, self.function_space)


        self.u = fe.Function(self.function_space)

        #self.compute_error()
        self.save_snapshot()

#==================================================================
    def save_snapshot(self):

        if self.fast_run:
            return

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        y   = self.u_n

        ax.plot(self.mesh, y, 'ko', linewidth=2, label='Approx.')

        auc = srule(y, self.mesh)

        eps = 1e-2

        txt = 't = ' + '{:0.3f}'.\
                format(self.current_time-self.initial_time) +\
                ', AUC: ' + '{:0.3f}'.format(auc)
        ax.text(0.1, 0.1, txt, fontsize=16)

        ax.legend(loc=1)

        fname = 'solution_' + str(self.counter) + self.figure_format
        fname = os.path.join(self.fem_solution_storage_dir, fname)
        self.counter += 1
        ax.set_ylim([-2-eps,1+eps])
        fig.savefig(fname, dpi=150)
        plt.close('all')


#==================================================================
    def lambda_m(self, m):
        return np.pi * m / self.L

#==================================================================
    def X(self, m, x):
        return np.cos(x * self.lambda_m(m))

#==================================================================
    def initial_condition(self, x):
        return 1/(1 + np.exp(30*(x-0.5)))

#==================================================================
    def IC(self, m):
        norm_sq = self.L
        if 0 < m:
            norm_sq /= 2
        x = np.linspace(0, self.L, 5000)
        y  = self.initial_condition(x) - self.shift(x)
        y *= self.X(m, x)
        auc  = srule(y, x)
        auc /= norm_sq
        return auc


#==================================================================
    def W(self, m, t=0):
        lm_sq_p1 = self.lambda_m(m)**2 + 1
        exp_term = np.exp(-t * lm_sq_p1) 
        return exp_term


#==================================================================
    def shift(self, x):
        return 1 - x * 0


#==================================================================
    def steady_state(self,x):
        #CHECK
        return 0


#==================================================================
    def U(self, x, t):
        s = self.shift(x)
        for i in range(0,self.n_components):
            increment = self.W(i, t) * self.IC(i) * self.X(i, x)
            s += increment
        return s

#==================================================================
    def plot_exact_solution(self, t = 1): 

        self.L = 5*np.pi/np.sqrt(3)
        self.time = 1.5

        mesh_density = 500
        x_mesh = np.linspace(0, self.L, mesh_density)
        y = self.U(x_mesh, t)
        ss= self.steady_state(x_mesh)
        eps = 1e-2
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(x_mesh, y, 'b-')
        #ax.plot(x_mesh, ss, 'r+')
        ax.set_xlim([self.x_left-eps,self.x_right+eps])
        ax.set_ylim([0-eps,1+eps])
        fig.savefig('sol.pdf', dpi=300)

#==================================================================
    def create_movie(self):

        rx = re.compile('(?P<number>\d+)')
        fnames = []
        fnumbers = []

        for f in os.listdir(self.fem_solution_storage_dir):
            if '.png' in f:
                obj = rx.search(f)
                if obj is not None:
                    number = int(obj.group('number'))
                    fnames.append(f)
                    fnumbers.append(number)

        fnumbers = np.array(fnumbers)
        indices = np.argsort(fnumbers)

        fnames = [fnames[x] for x in indices]
        images = []

        for i in range(len(fnames)):
            if i % 1 == 0 and i < 17:
                images.append(fnames[i])

        if len(images) == 0:
            print('Empty list of images')
            return

        if self.video_format == '.ogv':
            fourcc = cv2.VideoWriter_fourcc(*'THEO')

        elif self.video_format == '.mp4':
            fourcc = 0x7634706d

        elif self.video_format == '.webm':
            fourcc = cv2.VideoWriter_fourcc(*'VP80')

        video_name = 'simulation_' + self.label + '_' +\
                str(self.dimension) + 'D' + self.video_format
        video_fname = os.path.join(self.movie_dir, video_name)

        im_path = os.path.join(self.fem_solution_storage_dir, images[0])
        frame = cv2.imread(im_path)
        height, width, layers = frame.shape
        print(np.array(frame.shape)/2)
        video = cv2.VideoWriter(video_fname, fourcc, self.fps, (width, height))

        print('Creating movie:', video_name)
        for im in images:
            im_path = os.path.join(self.fem_solution_storage_dir, im)
            video.write(cv2.imread(im_path))

        cv2.destroyAllWindows()
        video.release()
        print('Finished creating movie')

#==================================================================
    def create_simple_mesh(self):

        print('Creating simple mesh')
        nx = ny = self.mesh_density

        if self.dimension == 1: 

            self.mesh = fe.IntervalMesh(nx, self.x_left, self.x_right)


        if self.dimension == 2: 
            self.mesh = fe.UnitSquareMesh(nx, ny)

        #self.plot_mesh()

#==================================================================
    def create_mesh(self):

        if self.mode == 'test' or self.dimension == 1:
            self.create_simple_mesh()
        else:
            self.create_coronal_section_mesh(self.model_z_n)

#==================================================================
    def create_coronal_section_mesh(self, model_z_n = 0):

        brain_slice = self.image_manipulation_obj.\
                get_brain_slice_from_model_z_n(model_z_n)

        x_size = 65
        self.boundary_points = brain_slice.\
                generate_boundary_points_ccw(x_size)
        domain_vertices = []

        for x,y in zip(self.boundary_points[0], self.boundary_points[1]):
            domain_vertices.append(fe.Point(x,y))

        geo         = mesher.Polygon(domain_vertices)
        self.mesh   = mesher.generate_mesh(geo, self.mesh_density);

        #self.plot_mesh()

#==================================================================
    def set_function_spaces(self):

        self.function_space = fe.FunctionSpace(self.mesh, 'P',\
                self.polynomial_degree)

#==================================================================
    def dofs_to_coordinates(self):

        if self.dimension != 1:
            return

        mesh_dim = self.mesh.geometry().dim()
        dof_coordinates = self.function_space.\
                tabulate_dof_coordinates().ravel()
        self.dof_map = np.argsort(dof_coordinates)
        self.ordered_mesh = dof_coordinates[self.dof_map]
        delta_x = self.ordered_mesh[1] - self.ordered_mesh[0]
        print('2^(',np.log2(delta_x), ') =', delta_x)

#==================================================================
    def create_bilinear_form_and_rhs(self):


        # Define variational problem
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)

        self.bilinear_form = (1 + self.lam * self.dt) *\
                (u * v * fe.dx) + self.dt * self.diffusion_coefficient *\
                (fe.dot(fe.grad(u), fe.grad(v)) * fe.dx)

        self.rhs = (self.u_n + self.dt * self.rhs_fun) * v * fe.dx

#==================================================================
    def solve_problem(self):

        '''
        LR_Neumann boundary conditions
        '''
        fe.solve(self.bilinear_form == self.rhs, self.u)

#==================================================================
    def compute_error(self):

        if self.mode != 'test':
            return

        error_L2 = fe.errornorm(self.boundary_fun, self.u_n, 'L2')
        error_LI = np.abs(\
                fe.interpolate(\
                self.boundary_fun,self.function_space).vector().get_local() -\
                self.u_n.vector().get_local()\
                ).max()

        print('L2 error at t = {:.3f}: {:.2e}'.format(\
                self.current_time, error_L2))

        print('LI error at t = {:.3f}: {:.2e}'.format(\
                self.current_time, error_LI))

        self.error_list.append(error_L2) 

#==================================================================
    def run(self):
        self.set_data_dirs()
        neumann_fun   = lambda t: 0
        ic_fun        = lambda x: 1/(1+np.exp(100*(x-0.05)))
        dirichlet_fun = lambda t: 0
        self.mesh     = np.linspace(0,self.L, self.mesh_density)
        self.current_time = 0
        self.u_n = ic_fun(self.mesh)
        self.save_snapshot()
        self.u   = self.u_n * 0
        dx = self.mesh[1]-self.mesh[0]
        dx_sq = dx * dx
        k  = self.dt/dx_sq

        print('dx',dx)
        print('dt',self.dt)
        print('k',k)
        while self.current_time < self.final_time: 
            
            self.current_time += self.dt
            print('t(', self.counter, ')= {:0.3f}'.format(self.current_time))
            self.u[-1] = dirichlet_fun(self.current_time)
            #print('u[',self.mesh_density-1,']=',self.u[-1])
            self.u[-2] = self.u[-1] - dx * neumann_fun(self.current_time)
            #print('u[',self.mesh_density-2,']=',self.u[-2])
            for i in range(self.mesh_density-2,0,-1):
                self.u[i-1] = (self.u[i] - self.u_n[i])/k -\
                        self.u[i+1] + 2*self.u[i] -\
                        dx_sq * (1 - self.u[i]) 
                #print('u[',i-1,']=', self.u[i-1])
            self.u_n = self.u + 0
            self.save_snapshot()
            print('------------------------')

        print('Alles ist gut')

#==================================================================
    def run_cn(self):
        self.set_data_dirs()
        neumann_fun   = lambda t: 0
        ic_fun        = lambda x: 1/(1+np.exp(100*(x-0.05)))
        dirichlet_fun = lambda t: 0
        self.mesh     = np.linspace(0,self.L, self.mesh_density)
        self.current_time = 0
        self.u_n = ic_fun(self.mesh)
        self.save_snapshot()
        self.u   = self.u_n * 0
        dx = self.mesh[1]-self.mesh[0]
        dx_sq = dx * dx
        k  = self.dt/dx_sq

        print('dx',dx)
        print('dt',self.dt)
        print('k',k)
        while self.current_time < self.final_time: 
            
            self.current_time += self.dt
            print('t(', self.counter, ')= {:0.3f}'.format(self.current_time))
            self.u[-1] = dirichlet_fun(self.current_time)
            #print('u[',self.mesh_density-1,']=',self.u[-1])
            self.u[-2] = self.u[-1] - dx * neumann_fun(self.current_time)
            #print('u[',self.mesh_density-2,']=',self.u[-2])
            for i in range(self.mesh_density-2,0,-1):
                self.u[i-1] = 2 * (self.u[i] - self.u_n[i])/k -\
                        (\
                        self.u[i+1] - 2*self.u[i] +\
                        self.u_n[i+1] - 2*self.u_n[i] + self.u_n[i-1] + \
                        dx_sq * (2 - (self.u[i] + self.u_n[i]))\
                        )
                print('u[',i-1,']=', self.u[i-1])
            self.u_n = self.u + 0
            self.save_snapshot()
            print('------------------------')

        print('Alles ist gut')


