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


#from scipy.optimize import least_squares as lsq
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from scipy.spatial import Delaunay as delaunay
from scipy.integrate import simps as srule

from image_manipulation_module import ImageManipulation

class TimeDamped():

#==================================================================
    def __init__(self):

        self.label        = 'TD'
        self.n_components = 0
        self.L            = 10
        self.time         = 0
        self.fast_run     = False
        self.model_z_n    = 535
        self.hole_radius  = 0.2

        #self.set_fem_properties()
        self.mode         = 'exp'
        self.dimension    = 2
        self.polynomial_degree = 2
        self.mesh_density = 64
        self.dt           = 0.01
        self.x_left       = 0
        self.x_right      = self.L
        self.fps          = 0

        self.image_manipulation_obj = ImageManipulation()

        self.initial_time    = 0.01

        if self.dimension == 1:
            self.final_time = 60
        if self.dimension == 2:
            self.final_time = 8

        self.current_time    = None
        self.counter         = 0
        self.boundary_points = None
        self.figure_format   = '.png'
        self.video_format    = '.webm'

        self.alpha        = 4.3590
        self.beta         = -0.156
        self.lam          = 0
        self.kappa        = 30

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
        self.geometry     = None

        self.current_dir  = os.getcwd()
        self.postprocessing_dir = os.path.join(self.current_dir,\
                'postprocessing')
        self.fem_solution_storage_dir = os.path.join(self.current_dir,\
                'solution', self.label, str(self.dimension) + 'D')
        self.movie_dir = os.path.join(self.fem_solution_storage_dir, 'movie')

        self.u_experimental = None
        self.residual_list  = None

        self.plot_true_solution= True
        self.plot_symmetric    = True

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

        self.rhs_fun = fe.Expression('D/(2*t)', degree=2,\
                t = 0.01, D = self.dimension)

#==================================================================
    def create_boundary_conditions(self):

        if self.dimension == 2:
            self.boundary_fun = fe.Constant(1)
            def is_on_the_boundary(x, on_boundary):
                return on_boundary and 4.34 < x[0]

        tol = 1e-6
        if self.dimension == 1:

            self.boundary_fun = fe.Expression(\
                    'exp(-x[0]*x[0]/(4*t))', degree=2, t=0.01)

            def is_on_the_boundary(x, on_boundary):
                return on_boundary 

        self.boundary_conditions = fe.DirichletBC(\
                self.function_space, self.boundary_fun,\
                is_on_the_boundary)

#==================================================================
    def set_initial_conditions(self):

        self.current_time = self.initial_time 

        if self.dimension == 2:
            self.ic_fun = fe.Expression(\
                    'exp(-kappa * (pow(x[0]-alpha,2) + pow(x[1]-beta,2)))',\
                    degree = 2,\
                    alpha  = self.alpha,\
                    beta   = self.beta,\
                    kappa  = self.kappa)
            self.u_n = fe.project(self.ic_fun, self.function_space)

        if self.dimension == 1:
            self.boundary_fun.t = self.current_time
            self.u_n = fe.interpolate(self.boundary_fun, self.function_space)


        self.u = fe.Function(self.function_space)

        #self.compute_error()
        self.save_snapshot()

#==================================================================
    def save_snapshot(self):

        if self.fast_run:
            return

        if self.dimension == 2:
            self.vtkfile << (self.u_n, self.current_time)
            self.counter += 1
            return

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        y   = self.u_n.vector().get_local()[self.dof_map]

        ax.plot(self.ordered_mesh, y, 'ko', linewidth=2, label='Approx.')

        auc = srule(y, self.ordered_mesh)

        if self.plot_symmetric:

                reflected_mesh = -self.ordered_mesh[::-1]
                ax.plot(reflected_mesh, y[::-1], 'ko', linewidth=2)
                auc *= 2


        eps = 1e-2

        if self.plot_true_solution:

            y_true = self.U(self.ordered_mesh, self.current_time)

            ax.plot(self.ordered_mesh, y_true,\
                    'b-', linewidth=2, label='Exact')

            if self.plot_symmetric:
                    ax.plot(reflected_mesh, y_true[::-1],\
                            'b-', linewidth=2)

            ax.set_ylim([-0-eps,1+eps])
            ax.set_xlim([self.x_left-eps, self.x_right+eps])

            if self.plot_symmetric:
                ax.set_xlim([-self.x_right-eps, self.x_right+eps])


        txt = 't = ' + '{:0.3f}'.\
                format(self.current_time-self.initial_time) +\
                ', AUC: ' + '{:0.3f}'.format(auc)
        ax.text(0.1, 0.1, txt, fontsize=16)

        ax.legend(loc=1)

        fname = 'solution_' + str(self.counter) + self.figure_format
        fname = os.path.join(self.fem_solution_storage_dir, fname)
        self.counter += 1
        fig.savefig(fname, dpi=150)
        plt.close('all')


#==================================================================
    def U(self, x, t):
        return np.exp(-x*x/(4 * t))

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
            if i % 10 == 0 and i < np.inf:
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
        print('(H,W) = ', np.array(frame.shape)/2)

        if self.dimension == 1:
            self.fps = 70
        if self.dimension == 2:
            self.fps = 8

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

        self.geometry = mesher.Polygon(domain_vertices)
        self.mesh     = mesher.generate_mesh(self.geometry, 16);

        self.puncture_mesh()

#==================================================================
    def generate_holes(self):

        fname = os.path.join(self.fem_solution_storage_dir, 'holes.txt')
        if os.path.exists(fname):
            print('Found a file containing the hole data')
            return np.loadtxt(fname)

        boundary = fe.BoundaryMesh(self.mesh, 'exterior')
        bbtree   = fe.BoundingBoxTree()
        bbtree.build(boundary)

        np.random.seed(123)
        N = 40
        max_n_tries = 1e6
        counter = 0
        L = []
        gap = 0.1

        while len(L) < N and counter < max_n_tries:

            counter += 1
            x = np.random.rand()*8 - 4
            y = np.random.rand()*4.8 - 2.4
            p = np.array([x,y])
            p_fenics = fe.Point(p)
            _, distance_to_boundary = bbtree.compute_closest_entity(p_fenics)

            rejected = False

            if distance_to_boundary < self.hole_radius + gap:
                continue

            for c in L: 
                if np.linalg.norm(c-p) < 2*self.hole_radius + gap:
                    rejected = True
                    break

            if not rejected:
                L.append(p)

        L = np.array(L)
        fname = os.path.join(self.fem_solution_storage_dir, 'holes.txt')
        np.savetxt(fname, L)

        print('Found', N, 'circles in', counter, 'trials')

        return L

#==================================================================
    def puncture_mesh(self): 

        L = self.generate_holes()

        for p in L:
            circle = mesher.Circle(fe.Point(p), self.hole_radius) 
            self.geometry -= circle

        self.mesh = mesher.generate_mesh(self.geometry, self.mesh_density);
        print('Done with the mesh generation')

#==================================================================
    def plot_mesh(self):

        fig = plt.figure()
        ax  = fig.add_subplot(111)

        for t in self.mesh.cells():
            points = self.mesh.coordinates()[t]

            if self.dimension == 2:
                indices = [0,1,2,0]
                ax.plot(points[indices,0], points[indices,1], 'b-')

            elif self.dimension == 1:
                ax.plot(points[:,0], points[:,0]*0, 'b-')


            #for index, p in zip(t,points):
                #ax.text(p[0],p[1],str(index),fontsize=8)

        #bmesh    = fe.BoundaryMesh(self.mesh, 'exterior', True)
        #b_points = bmesh.coordinates()

        fname = os.path.join(self.postprocessing_dir, 'mesh.png')
        fig.savefig(fname, dpi=300)

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

        self.rhs = (self.u_n + self.dt * self.rhs_fun * self.u_n) * v * fe.dx

#==================================================================
    def solve_problem(self):

        '''
        Boundary conditions
        '''
        fe.solve(self.bilinear_form == self.rhs,\
                self.u, self.boundary_conditions)

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
        self.create_rhs_fun()
        self.create_mesh()
        self.set_function_spaces()
        self.dofs_to_coordinates()
        self.create_boundary_conditions()
        self.set_initial_conditions()
        self.create_bilinear_form_and_rhs()

        while self.current_time < self.final_time: 
            
            self.current_time += self.dt
            print('t(', self.counter, ')= {:0.3f}'.format(self.current_time))
            self.boundary_fun.t = self.current_time
            self.rhs_fun.t      = self.current_time
            self.solve_problem()
            self.u_n.assign(self.u)
            self.compute_error()
            self.save_snapshot()
            print('------------------------')
            
        
        print('Alles ist gut')


