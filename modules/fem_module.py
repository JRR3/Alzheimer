#==========================================
# Houston Methodist Research Institute
# May 21, 2019
# Supervisor: Vittorio Cristini
# Developer : Javier Ruiz Ramirez
#==========================================

import fenics as fe
import numpy as np
import mshr as mesher
import sympy 
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
#from analytical_solution_1d_dd_module import AnalyticalSolution1D as Exact

#==================================================================

class FEMSimulation():

#==================================================================
    def __init__(self, storage_dir = None):

        self.current_dir  = os.getcwd()

        self.postprocessing_dir = os.path.join(self.current_dir,\
                'postprocessing')

        self.fast_run          = None
        self.mode              = 'exp'
        self.model_z_n         = 535
        self.dimension         = 1
        self.polynomial_degree = 2
        self.mesh_density      = 64
        self.dt                = 0.005
        self.domain_length     = 250
        self.x_left            = 0
        self.x_right           = +self.domain_length
        self.plot_true_solution= True
        self.use_nonlinear     = True
        self.plot_symmetric    = False
        self.fps               = 30


        #self.exact_solution    = Exact(self.domain_length)
        self.image_manipulation_obj = ImageManipulation()

        self.initial_time    = 0
        self.final_time      = 2.0
        self.current_time    = None
        self.counter         = 0
        self.boundary_points = None
        self.figure_format   = '.png'
        #self.video_format    = '.mp4'
        #self.video_format    = '.ogv'
        self.video_format    = '.webm'

        self.diffusion_coefficient = 1.0
        self.lam                   = 1

        self.error_list   = []
        self.u_exact_str  = None
        self.u_true       = None
        self.rhs_fun_str  = None
        self.alpha        = 0
        self.beta         = 0
        self.kappa        = 0
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

        self.fem_solution_storage_dir = os.path.join(self.current_dir,\
                'solution', str(self.dimension) + 'D')

        self.u_experimental = None
        self.residual_list  = None


        self.set_data_dirs()

#==================================================================
    def plot_coordinate_map(self):

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        x = np.array([-4,-2,0,+0.6])
        y = np.array([-1.1, -2.495, -4.295, -4.895])
        ax.plot(x,y,'b-')
        ax.plot(x[[0,-1]],y[[0,-1]],'r-')
        slope = (y[-1] - y[0])/(x[-1]-x[0])
        b     = -slope * x[0] + y[0]
        print('Line equation: y =', slope, '* x +', b)
        ax.set_aspect('equal')
        ax.set_xlabel("Giulia's coordinates")
        ax.set_ylabel("Model's coordinates")
        fname = os.path.join(self.postprocessing_dir, 'coordinate_map.pdf')
        fig.savefig(fname, dpi=300)

#==================================================================
    def full_brain_mesh(self):
        path_dir = os.path.join(self.current_dir,\
                'raw_data/Tumor-bearing/HG-NPs_tumor-bearing/1D/storage')
        regexp = re.compile(r'mesh_coordinates_([0-9]+)[.]txt')
        L = []
        counter = 0
        for f in os.listdir(path_dir):
            obj = regexp.match(f)
            if obj is None:
                continue
            fname = os.path.join(path_dir, f)
            m = np.loadtxt(fname)
            m = np.unique(m,axis=0)
            L.append(m)
            counter += 1

        m = np.vstack(L)
        fname = os.path.join(path_dir, 'full_brain_mesh_points.txt')
        np.savetxt(fname, m)
        triangulation = delaunay(m)
        fname = os.path.join(path_dir, 'full_brain_mesh_simplices.txt')
        simplices = triangulation.simplices
        np.savetxt(fname, simplices, fmt='%d')

        '''
        Plot
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(m[:,0],m[:,1],m[:,2], triangles=simplices)
        ax.set_aspect('equal')
        fname = os.path.join(path_dir, 'trisurf.pdf')
        fig.savefig(fname, dpi=300)

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

        fname = os.path.join(self.postprocessing_dir, 'mesh.pdf')
        fig.savefig(fname, dpi=300)


#==================================================================
    def set_data_dirs(self):

        if not os.path.exists(self.postprocessing_dir):
            os.makedirs(self.postprocessing_dir)

        if not os.path.exists(self.fem_solution_storage_dir):
            os.makedirs(self.fem_solution_storage_dir)

        txt = 'solution.pvd'
        fname = os.path.join(self.fem_solution_storage_dir, txt)
        self.vtkfile = fe.File(fname)

#==================================================================
    def set_parameters(self):

        if self.mode == 'test': 
            self.alpha        = 3.0
            self.beta         = 1.2
            self.lam          = 1.0


        else:

            if self.dimension == 1:

                self.alpha        = 0
                self.beta         = 0
                #self.lam          = 1.0
                self.lam          = 0.0
                self.kappa        = 30

                if self.use_nonlinear: 

                    self.initial_time    = 4.4
                    self.final_time      = 126
                    self.dt              = 0.25

                    #self.initial_time    = 0.01
                    #self.final_time      = 40
                    #self.dt              = 0.01

            elif self.dimension == 2:

                self.alpha        = 4.3590
                self.beta         = -0.156
                self.lam          = 1.0
                self.kappa        = 30

        print('Lambda: ', self.lam)
        print('Alpha : ', self.alpha)
        print('Beta  : ', self.beta)
        print('Kappa : ', self.kappa)


#==================================================================
    def create_initial_condition_function(self):

        if self.mode == 'test':
            return

        x,y,a,b,k = sympy.symbols('x[0], x[1], alpha, beta, kappa')

        if self.dimension == 1: 
            ic = sympy.exp(-k * ((x-a)**2))

        if self.dimension == 2: 
            ic = sympy.exp(-k * ((x-a)**2 + (y-b)**2))

        ic_str = sympy.printing.ccode(ic)

        self.ic_fun =\
                fe.Expression(ic_str, degree=2,\
                alpha = self.alpha, beta = self.beta, kappa = self.kappa)

#==================================================================
    def create_exact_solution_and_rhs_fun_strings(self):

        if self.mode != 'test':
            return

        print('Creating exact solution and rhs strings')

        x,y,a,b,l,t = sympy.symbols('x[0], x[1], alpha, beta, lam, t')

        if self.dimension == 2: 
            u_exact = 1 + x**2 + a * y**2 + b * t

        if self.dimension == 1: 
            u_exact = 1 + a * x**2 + b * t
            self.u_true = lambda xx,tt: 1 + self.alpha * xx**2 + self.beta * tt

        u_t = u_exact.diff(t)

        if self.dimension == 1: 
            grad_u = u_exact.diff(x)
            diffusion_grad_u = self.diffusion_coefficient * grad_u
            diffusion_term = diffusion_grad_u.diff(x)

        if self.dimension == 2: 
            grad_u = sympy.Matrix([u_exact]).jacobian([x,y]).T
            diffusion_grad_u = self.diffusion_coefficient * grad_u
            diffusion_term = diffusion_grad_u.jacobian([x,y]).trace()

        rhs_fun = u_t - diffusion_term + l*u_exact

        self.u_exact_str = sympy.printing.ccode(u_exact)
        self.rhs_fun_str = sympy.printing.ccode(rhs_fun)

#==================================================================
    def create_rhs_fun(self):

        if self.mode == 'test': 

            print('Creating rhs function')
            self.rhs_fun = fe.Expression(self.rhs_fun_str, degree=2,\
                    alpha = self.alpha,\
                    beta  = self.beta,\
                    lam   = self.lam,\
                    t     = 0)
        else:
            '''
            Constant RHS for the experimental case
            '''
            self.rhs_fun = fe.Constant(0)
            self.rhs_fun = fe.Expression('1/(2*t)', degree=2, t = 0.01)

#==================================================================
    def create_boundary_conditions(self):

        if self.mode == 'test':

            print('Creating boundary function')
            self.boundary_fun = fe.Expression(self.u_exact_str, degree=2,\
                    alpha = self.alpha, beta = self.beta, t = 0)

        else:
            '''
            Constant boundary conditions
            '''


        if self.dimension == 2:
            self.boundary_fun = fe.Constant(1)
            def is_on_the_boundary(x, on_boundary):
                return on_boundary and 4.34 < x[0]

        tol = 1e-6
        if self.dimension == 1:

            self.boundary_fun = fe.Expression('x[0] < 1e-6 ? 1 : 0', degree=0)

            c    = -5 / np.sqrt(6)
            txt = 'pow(1+exp((x[0]+C*t)/sqrt(6)), -2)'
            self.boundary_fun = fe.Expression(txt, degree=2, C=c, t=0)

            #self.boundary_fun = fe.Expression(\
                    #'exp(-x[0]*x[0]/(4*t))', degree=2, t=0.01)

            def is_on_the_boundary(x, on_boundary):
                return on_boundary
                #return on_boundary and x[0] < 1e-6
                #return False

        self.boundary_conditions = fe.DirichletBC(\
                self.function_space, self.boundary_fun, is_on_the_boundary)

#==================================================================
    def V(self, z):
        return np.power(1+1*np.exp(z / np.sqrt(6)), -2.)

    def tw(self, x, t):
        c = -5 / np.sqrt(6)
        return self.V(x + c*t)

    def smg(self, x, t):
        return np.exp(-(x**2)/(4*t))

#==================================================================
    def plot_smg(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        t = 40.1
        x = np.linspace(-10,10,5000)
        ax.plot(x, self.smg(x,t),'b-')
        ax.set_ylim([0-1e-3,1+1e-3])
        fig.savefig(os.path.join(self.fem_solution_storage_dir,'a.pdf'))

#==================================================================
    def plot_traveling_wave(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        t = 8
        x = np.linspace(-10,10,5000)
        ax.plot(x, self.tw(x,t),'b-')
        ax.set_ylim([0-1e-3,1+1e-3])
        fig.savefig(os.path.join(self.fem_solution_storage_dir,'a.pdf'))


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
    def set_initial_conditions(self):

        self.current_time = self.initial_time

        #Initial condition
        #self.u_n = fe.project(self.boundary_fun, self.function_space)

        if self.mode == 'test':
            print('Setting initial conditions')
            self.boundary_fun.t = self.current_time
            self.u_n = fe.interpolate(self.boundary_fun, self.function_space)

        else:

            if self.dimension == 2:
                self.u_n = fe.project(self.ic_fun, self.function_space)

            if self.dimension == 1:
                #self.u_n = fe.interpolate(fe.Constant(0), self.function_space)
                #self.u_n.vector()[self.dof_map[0]] = 1
                self.boundary_fun.t = self.current_time
                self.u_n = fe.interpolate(self.boundary_fun, self.function_space)

        self.u = fe.Function(self.function_space)

        self.compute_error()
        self.save_snapshot()

#==================================================================
    def create_bilinear_form_and_rhs(self):


        # Define variational problem
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)

        self.bilinear_form = (1 + self.lam * self.dt) *\
                (u * v * fe.dx) + self.dt * self.diffusion_coefficient *\
                (fe.dot(fe.grad(u), fe.grad(v)) * fe.dx)

        self.rhs = (\
                self.u_n +\
                0 * self.dt * self.rhs_fun * self.u_n +\
                1 * self.use_nonlinear * self.dt * (1-self.u_n) * self.u_n\
                ) * v * fe.dx

#==================================================================
    def solve_problem(self):

        '''
        Dirichlet boundary conditions
        '''
        fe.solve(self.bilinear_form == self.rhs,\
                self.u, self.boundary_conditions)


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

        if self.plot_symmetric:
                reflected_mesh = -self.ordered_mesh[::-1]
                ax.plot(reflected_mesh, y[::-1], 'ko', linewidth=2)

        auc = srule(y,self.ordered_mesh)

        if self.mode == 'test':
            y_true = self.u_true(self.ordered_mesh, self.current_time)
            ax.plot(self.ordered_mesh, y_true, 'ro')
            ax.set_ylim([0,6])

        else:
            eps = 1e-2

            if self.plot_true_solution:

                if self.use_nonlinear:
                    y_true = self.tw(\
                            self.ordered_mesh, self.current_time)

                    #y_true = self.smg(\
                            #self.ordered_mesh, self.current_time)
                else:
                    y_true = self.exact_solution.U(\
                            self.ordered_mesh, self.current_time)

                ax.plot(self.ordered_mesh, y_true,\
                        'b-', linewidth=2, label='Exact')

                if self.plot_symmetric:
                        ax.plot(reflected_mesh, y_true[::-1],\
                                'b-', linewidth=2)

            ax.set_ylim([0-eps,1+eps])
            ax.set_xlim([self.x_left-eps, self.x_right+eps])

            if self.plot_symmetric:
                ax.set_xlim([-self.x_right-eps, self.x_right+eps])


        txt = 't = ' + '{:0.3f}'.format(self.current_time-self.initial_time) +\
                ', AUC: ' + '{:0.3f}'.format(auc)
        ax.text(0.1, 0.1, txt, fontsize=16)

        ax.legend(loc=1)

        fname = 'solution_' + str(self.counter) + self.figure_format
        fname = os.path.join(self.fem_solution_storage_dir, fname)
        self.counter += 1
        fig.savefig(fname, dpi=150)
        plt.close('all')


#==================================================================
    def system_setup(self):

        self.create_mesh()
        self.set_function_spaces()
        self.create_rhs_fun()
        self.create_boundary_conditions()
        self.load_opt_experimental_data()


#==================================================================
    def dofs_to_coordinates(self):

        if self.dimension != 1:
            return

        mesh_dim = self.mesh.geometry().dim()
        dof_coordinates = self.function_space.\
                tabulate_dof_coordinates().ravel()
        self.dof_map = np.argsort(dof_coordinates)
        self.ordered_mesh = dof_coordinates[self.dof_map]
        self.delta_x = self.ordered_mesh[1] - self.ordered_mesh[0]
        print('2^(',np.log2(self.delta_x), ') =', self.delta_x)

#==================================================================
    def create_movie(self):

        if self.dimension == 1:
            movie_dir = self.fem_solution_storage_dir

        if self.dimension == 2:
            movie_dir = os.path.join(self.fem_solution_storage_dir, 'movie')

        rx = re.compile('(?P<number>\d+)')
        fnames = []
        fnumbers = []

        for f in os.listdir(movie_dir):
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
            if i % 1 == 0 and i < np.inf:
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

        video_name = 'simulation_' +\
                str(self.dimension) + 'D' + self.video_format
        video_fname = os.path.join(movie_dir, video_name)
        im_path = os.path.join(movie_dir, images[0])
        frame = cv2.imread(im_path)
        height, width, layers = frame.shape
        print(frame.shape)
        video = cv2.VideoWriter(video_fname, fourcc, 30, (width, height))

        print('Creating movie:', video_name)
        for im in images:
            im_path = os.path.join(movie_dir, im)
            video.write(cv2.imread(im_path))

        cv2.destroyAllWindows()
        video.release()
        print('Finished creating movie')



#==================================================================
    def run(self):

        #self.set_parameters()
        #self.create_exact_solution_and_rhs_fun_strings()
        #self.create_initial_condition_function()
        #self.create_rhs_fun()
        self.create_mesh()
        self.set_function_spaces()
        self.dofs_to_coordinates()
        #self.create_boundary_conditions()
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
        



