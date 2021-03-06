#==========================================
# Houston Methodist Research Institute
# May 21, 2019
# Supervisor: Vittorio Cristini
# Developer : Javier Ruiz Ramirez
#==========================================

import fenics as fe
import numpy as np
import mshr as mesher
import os
import re
import cv2


#from scipy.optimize import least_squares as lsq
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from scipy.spatial import Delaunay as delaunay
from scipy.integrate import simps as srule
from scipy.integrate import odeint as ode_solve

from image_manipulation_module import ImageManipulation

class TravelingWave():

#==================================================================
    def __init__(self):

        self.label        = 'TW'
        self.n_components = 0
        self.L            = 2
        self.time         = 0
        self.fast_run     = False
        self.model_z_n    = 535
        self.use_previously_stored_mesh = False
        self.n_columns    = 20
        self.keep_only_network = False



        #self.set_fem_properties()
        self.mode         = 'exp'
        self.dimension    = 3
        self.polynomial_degree = 1
        self.mesh_density = 16
        self.x_left       = 0
        self.x_right      = self.L
        self.fps          = 0

        self.n_squares       = 9
        self.square_fraction = 0.25
        self.hole_coordinates       = None
        self.hole_radius            = 0.2
        self.hole_fractional_volume = 0.60
        self.n_holes                = 80
        self.use_available_hole_data=False
        self.original_volume        = 0
        self.enforce_radius_of_holes = True

        self.image_manipulation_obj = ImageManipulation()

        if self.dimension == 1:
            self.dt           = 0.25
            self.initial_time = 4.4
            self.final_time   = 126

        elif self.dimension == 2:
            self.dt           = 0.01
            self.initial_time = 0
            self.final_time   = 8

        elif self.dimension == 3:
            self.dt           = 0.01
            self.initial_time = 0
            self.final_time   = 5*self.dt

        self.current_time    = 0
        self.counter         = 0
        self.boundary_points = None
        self.figure_format   = '.png'
        self.video_format    = '.webm'

        if self.dimension == 2:
            self.alpha        = 4.3590
            self.beta         = -0.156
            self.lam          = 0
            self.kappa        = 30

        elif self.dimension == 3:
            if self.keep_only_network: 
                self.alpha        = 0.1666666666
                self.beta         = 0.0555555555
                self.gamma        = 0
                self.lam          = 0
                self.kappa        = 50
            else:
                self.alpha        = 0
                self.beta         = 0
                self.gamma        = 0
                self.lam          = 0
                self.kappa        = 50

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

        if self.dimension == 3:

            self.fem_solution_storage_dir =\
                    os.path.join(self.fem_solution_storage_dir,\
                    str(self.n_columns) + '_columns')

            self.mesh_fname =\
                    os.path.join(self.fem_solution_storage_dir,\
                    'porous_mesh_' + str(self.n_columns) + '_columns.xml')

        self.movie_dir = os.path.join(self.fem_solution_storage_dir, 'movie')
        self.vtk_dir   = os.path.join(self.fem_solution_storage_dir, 'vtk')

        self.slope_field_dir = os.path.join(self.current_dir,\
                'solution', self.label, 'slope_field')

        self.slope_field_movie_dir = os.path.join(\
                self.slope_field_dir, 'movie')

        self.potential_dir = os.path.join(self.current_dir,\
                'solution', self.label, 'potential')

        self.potential_movie_dir = os.path.join(\
                self.potential_dir, 'movie')


        self.u_experimental = None
        self.residual_list  = None

        self.plot_true_solution= True
        self.plot_symmetric    = False

#==================================================================
    def plot_mesh(self):

        txt = 'pow(x[0]-0.5,2) + pow(x[1]-0.5,2) + pow(x[2]-0.5,2)'
        f = fe.Expression(txt, degree=2)
        self.u_n = fe.interpolate(f, self.function_space)
        self.save_snapshot()

#==================================================================
    def set_data_dirs(self):

        if not os.path.exists(self.postprocessing_dir):
            os.makedirs(self.postprocessing_dir)

        if not os.path.exists(self.fem_solution_storage_dir):
            os.makedirs(self.fem_solution_storage_dir)

        if not os.path.exists(self.movie_dir):
            os.makedirs(self.movie_dir)

        if not os.path.exists(self.vtk_dir):
            os.makedirs(self.vtk_dir)

        if not os.path.exists(self.slope_field_dir):
            os.makedirs(self.slope_field_dir)

        if not os.path.exists(self.slope_field_movie_dir):
            os.makedirs(self.slope_field_movie_dir)

        if not os.path.exists(self.potential_dir):
            os.makedirs(self.potential_dir)

        if not os.path.exists(self.potential_movie_dir):
            os.makedirs(self.potential_movie_dir)

        txt   = 'solution.pvd'
        fname = os.path.join(self.vtk_dir, txt)
        self.vtkfile = fe.File(fname)

#==================================================================
    def create_rhs_fun(self):

        self.rhs_fun = fe.Constant(0)

#==================================================================
    def create_boundary_conditions(self):

        print('Creating boundary conditions')

        tol = 1e-6

        if self.dimension == 3:
            self.boundary_fun = fe.Constant(1)

            if self.keep_only_network:
                def is_on_the_boundary(x, on_boundary):
                    return on_boundary and\
                            x[2] < tol and\
                            x[1] < 0.1 and\
                            x[0] < 0.25
            else:
                def is_on_the_boundary(x, on_boundary):
                    return on_boundary and\
                            x[2] < tol and\
                            x[1] < 0.1 and\
                            x[0] < 0.1

        if self.dimension == 2:
            self.boundary_fun = fe.Constant(1)
            def is_on_the_boundary(x, on_boundary):
                return on_boundary and 4.34 < x[0]

        if self.dimension == 1:

            c    = -5 / np.sqrt(6)
            txt = 'pow(1+exp((x[0]+C*t)/sqrt(6)), -2)'
            self.boundary_fun = fe.Expression(txt, degree=2, C=c, t=0)

            def is_on_the_boundary(x, on_boundary):
                return on_boundary 

        self.boundary_conditions = fe.DirichletBC(\
                self.function_space, self.boundary_fun,\
                is_on_the_boundary)

#==================================================================
    def set_initial_conditions(self):

        print('Creating initial conditions')
        self.current_time = self.initial_time 

        if self.dimension == 3:
            txt = 'exp(-kappa *'  +\
                    '(pow(x[0]-alpha,2) +' +\
                    'pow(x[1]-beta,2) + pow(x[2]-gamma,2)))'
            self.ic_fun = fe.Expression(txt,\
                    degree = 2,\
                    alpha  = self.alpha,\
                    beta   = self.beta,\
                    gamma  = self.gamma,\
                    kappa  = self.kappa)

            self.u_n = fe.project(self.ic_fun, self.function_space)

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
        print('Storing initial state')
        self.save_snapshot()

#==================================================================
    def save_snapshot(self):

        if self.fast_run:
            return

        if 1 < self.dimension:
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

        auc = srule(y, self.ordered_mesh)

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
    def lambda_m(self, m):
        return np.pi * m / self.L

#==================================================================
    def W(self, z):
        return np.power(1+1*np.exp(z / np.sqrt(6)), -2.)


#==================================================================
    def U(self, x, t):
        c = -5 / np.sqrt(6)
        return self.W(x + c*t)

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
    def create_movie(self,\
            video_name = 'simulation',\
            img_dir    = None,\
            movie_dir  = None,\
            fps        = None): 

        if img_dir is None: 
            img_dir = self.movie_dir

        if movie_dir is None: 
            movie_dir = self.movie_dir

        rx = re.compile('(?P<number>\d+)')
        fnames = []
        fnumbers = []

        for f in os.listdir(img_dir):
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

        video_name = video_name + '_' + self.label + '_' +\
                str(self.dimension) + 'D' + self.video_format
        video_fname = os.path.join(movie_dir, video_name)

        im_path = os.path.join(img_dir, images[0])
        frame = cv2.imread(im_path)
        height, width, layers = frame.shape
        print('(H,W) = ', np.array(frame.shape)/2)

        if self.dimension == 1:
            self.fps = 30

        elif self.dimension == 2:
            self.fps = 30

        elif self.dimension == 3:
            self.fps = 30

        if fps is not None:
            self.fps = fps

        video = cv2.VideoWriter(video_fname, fourcc, self.fps, (width, height))

        print('Creating movie:', video_name)
        for im in images:
            im_path = os.path.join(img_dir, im)
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

        elif self.dimension == 2:
            self.create_coronal_section_mesh(self.model_z_n)

        elif self.dimension == 3:
            self.create_cocontinuous_mesh()

#==================================================================
    def create_base_net(self):

        square_jump = 1/self.n_squares
        r = square_jump * self.square_fraction

        y_square_jump = np.array([0, square_jump, 0])
        x_square_jump = np.array([square_jump, 0, 0]) 
        geo = None


        '''Lower left corner'''
        p = np.zeros(3) + 0.
        top    = p + 2*y_square_jump
        bottom = p + 2*x_square_jump 

        while (self.point_is_inside(p + 2*y_square_jump)):

            cylinder = mesher.Cylinder(fe.Point(top), fe.Point(bottom), r, r)
            if geo is None:
                geo = cylinder
            else:
                geo += cylinder

            p      += 2*y_square_jump
            top    += 2*y_square_jump
            bottom += 2*x_square_jump 





        '''Upper right corner'''
        p = self.L * np.ones(3)
        p[-1] = 0.
        top    = p - 2*y_square_jump
        bottom = p - 2*x_square_jump 

        while (self.point_is_inside(p - 2*y_square_jump)):

            cylinder = mesher.Cylinder(fe.Point(top), fe.Point(bottom), r, r)
            geo += cylinder

            p      -= 2*y_square_jump
            top    -= 2*y_square_jump
            bottom -= 2*x_square_jump 





        '''Lower Right corner'''
        p = np.zeros(3)
        p[0] = self.L
        top    = p + 2*y_square_jump
        bottom = p - 2*x_square_jump 

        while (self.point_is_inside(p + 2*y_square_jump)):

            cylinder = mesher.Cylinder(fe.Point(top), fe.Point(bottom), r, r)
            geo += cylinder

            p      += 2*y_square_jump
            top    += 2*y_square_jump
            bottom -= 2*x_square_jump 





        '''Upper left corner'''
        p = np.zeros(3)
        p[1] = self.L
        top    = p - 2*y_square_jump
        bottom = p + 2*x_square_jump 

        while (self.point_is_inside(p - 2*y_square_jump)):

            cylinder = mesher.Cylinder(fe.Point(top), fe.Point(bottom), r, r)
            geo += cylinder

            p      -= 2*y_square_jump
            top    -= 2*y_square_jump
            bottom += 2*x_square_jump 

        print('Finished creating base net')
        return geo

#==================================================================
    def extrude_base_net(self):

        net = self.create_base_net()

        square_jump = 1/self.n_squares
        r = square_jump * self.square_fraction
        n_levels = 3
        delta = self.L/(n_levels + 1)
        z_jump = np.array([0,0,delta])

        p = np.zeros(3)
        geo = None

        p += z_jump
        for k in range(n_levels):
            if geo is None:
                geo = mesher.CSGTranslation(net, fe.Point(p))
            else:
                geo += mesher.CSGTranslation(net, fe.Point(p))
            p += z_jump

        print('Finished creating extruded mesh')
        return geo

#==================================================================
    def create_cocontinuous_mesh(self):

        if self.use_previously_stored_mesh:

            self.mesh = fe.Mesh(self.mesh_fname)
            print('Mesh has been loaded from file')
            return

        p = self.L * np.ones(3)
        geo     = mesher.Box(fe.Point(0,0,0), fe.Point(p))
        net     = self.extrude_base_net()
        columns = self.create_cylindrical_columns()
        net    += columns

        if self.keep_only_network: 
            geo  = net
        else:
            geo -= net

        print('Finished creating the geometry')
        self.mesh = mesher.generate_mesh(geo, self.mesh_density);

        print('Writing mesh to file')
        mesh_file = fe.File(self.mesh_fname)
        mesh_file << self.mesh
        print('Finished writing mesh to file')


#==================================================================
    def create_base_grid_using_points(self):

        L = []
        z0= 0.

        square_jump = 1/self.n_squares
        right_square_jump = np.array([+square_jump, square_jump, 0])
        left_square_jump  = np.array([-square_jump, square_jump, 0])

        r = square_jump * self.square_fraction

        n_jumps = 2
        jump    = square_jump/(2 * n_jumps)

        left_up   = jump * np.array([-1,+1,0])
        left_down = jump * np.array([-1,-1,0])
        right_down= jump * np.array([+1,-1,0])
        right_up  = jump * np.array([+1,+1,0])


        #Negative slope
        reference_point  = np.array([0, 0, z0])
        reference_point += right_square_jump

        while (self.point_is_inside(reference_point)):
            L.append(reference_point + 0)

            p = reference_point + 0

            #Left Up
            p += left_up
            while (self.point_is_inside(p)):
                L.append(p+0)
                p += left_up

            p = reference_point + 0

            #Right down 
            p += right_down
            while (self.point_is_inside(p)):
                L.append(p+0)
                p += right_down


            reference_point += right_square_jump

        '''---------------------------------------'''
        #Positive slope
        reference_point  = np.array([1,0,z0])
        reference_point += left_square_jump

        while (self.point_is_inside(reference_point)):
            L.append(reference_point + 0)

            p = reference_point + 0

            #Right Up
            p += right_up
            while (self.point_is_inside(p)):
                L.append(p+0)
                p += right_up

            p = reference_point + 0

            #Left down 
            p += left_down
            while (self.point_is_inside(p)):
                L.append(p+0)
                p += left_down

            reference_point += left_square_jump

#==================================================================
    def create_cylindrical_columns(self):

        B = []
        square_jump = 1/self.n_squares
        r = square_jump * self.square_fraction

        short_jump = np.array([square_jump/2, 0, 0])
        long_jump  = np.array([square_jump, 0, 0])

        reference_point = np.array([square_jump/2, square_jump/2, 0])
        y_jump = np.array([0, square_jump/2, 0])

        state = 2

        while(self.point_is_inside(reference_point)):

            p = reference_point + 0

            if state % 2 != 0:
                p_jump = short_jump
            else:
                p_jump = long_jump

            if state % 4 != 0:
                p += p_jump

            state += 1
            inner_state = 0

            while(self.point_is_inside(p)):

                if np.array_equal(p_jump, short_jump):
                    break

                if inner_state % 2 == 0:
                    B.append(p+0)

                p += p_jump
                inner_state += 1

            reference_point += y_jump


        B = np.array(B)
        geo = None

        counter = 0
        n_seeds = len(B)
        chosen_columns = []
        np.random.seed(123)

        while counter < self.n_columns:

            if counter == 0:
                '''Necessary for initial and boundary conditions'''
                rand_int = 0
            else:
                rand_int = np.random.randint(0, n_seeds)

            if rand_int in chosen_columns:
                continue

            p = B[rand_int]
            chosen_columns.append(rand_int)
            counter += 1
            top     = p + 0
            top[-1] = 1
            bottom  = p + 0
            bottom[-1] = 0
            cylinder = mesher.Cylinder(fe.Point(top), fe.Point(bottom), r, r)

            if geo is None:
                geo = cylinder
            else:
                geo += cylinder

        print('Finished creating', counter, 'columns')
        return geo



#==================================================================
    def point_is_inside(self, p):
        eps = 1e-2
        return np.all(-eps < p) and np.all(p < self.L+eps)

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

        self.compute_mesh_volume()
        self.original_volume = self.mesh_volume
        print('Original volume', self.original_volume)
        self.generate_holes_2d()
        #self.compute_hole_fractional_volume()
        self.puncture_mesh()
        self.compute_mesh_volume()
        domain_volume = self.mesh_volume
        print('New volume', domain_volume)
        self.hole_fractional_volume = 1-domain_volume/self.original_volume
        print('Hole fractional volume:', self.hole_fractional_volume)
        print('# holes:', self.n_holes)
        print('Estimated hole fractional volume:',\
                self.compute_hole_fractional_volume())

#==================================================================
    def sphere_volume(self, r):

        factor = np.pi

        if self.dimension == 3:
            factor *= 4/3

        volume = factor * np.power(r, self.dimension)

        return volume

#==================================================================
    def compute_hole_fractional_volume(self):

        N = self.hole_coordinates.shape[0]
        empty_space = N * self.sphere_volume(self.hole_radius)

        return empty_space / self.original_volume

#==================================================================
    def compute_mesh_volume(self):

        one = fe.Constant(1)
        DG  = fe.FunctionSpace(self.mesh, 'DG', 0)
        v   = fe.TestFunction(DG)
        L   = v * one * fe.dx
        b   = fe.assemble(L)
        self.mesh_volume = b.get_local().sum()

#==================================================================
    def generate_holes_3d(self):

        fname = os.path.join(self.fem_solution_storage_dir, 'holes.txt')
        if self.use_available_hole_data and os.path.exists(fname):
            print('Found a file containing the hole data')
            self.hole_coordinates = np.loadtxt(fname)
            return

        boundary = fe.BoundaryMesh(self.mesh, 'exterior')
        bbtree   = fe.BoundingBoxTree()
        bbtree.build(boundary)

        np.random.seed(123)

        N = self.n_holes

        print('Hole radius:', self.hole_radius)
        gap = 0.07
        max_n_tries = 1e7
        counter = 0
        L = []

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

        self.hole_coordinates = np.array(L)
        fname = os.path.join(self.fem_solution_storage_dir, 'holes.txt')
        np.savetxt(fname, L)

        print('Found', N, 'circles in', counter, 'trials')

#==================================================================
    def generate_holes_2d(self):

        fname = os.path.join(self.fem_solution_storage_dir, 'holes.txt')
        if self.use_available_hole_data and os.path.exists(fname):
            print('Found a file containing the hole data')
            self.hole_coordinates = np.loadtxt(fname)
            return

        boundary = fe.BoundaryMesh(self.mesh, 'exterior')
        bbtree   = fe.BoundingBoxTree()
        bbtree.build(boundary)

        np.random.seed(123)

        N = self.n_holes

        if not self.enforce_radius_of_holes:
            self.hole_radius = np.sqrt(self.hole_fractional_volume *\
                    self.mesh_volume / np.pi / N)

        print('Hole radius:', self.hole_radius)
        gap = 0.15
        max_n_tries = 1e7
        counter = 0
        L = []

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

        self.hole_coordinates = np.array(L)
        fname = os.path.join(self.fem_solution_storage_dir, 'holes.txt')
        np.savetxt(fname, L)

        print('Found', N, 'circles in', counter, 'trials')


#==================================================================
    def puncture_mesh(self): 

        for p in self.hole_coordinates:
            circle = mesher.Circle(fe.Point(p), self.hole_radius) 
            self.geometry -= circle

        self.mesh = mesher.generate_mesh(self.geometry, self.mesh_density);
        print('Done with the mesh generation')

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


        print('Creating bilinear form')

        # Define variational problem
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)

        self.bilinear_form = (1 + self.lam * self.dt) *\
                (u * v * fe.dx) + self.dt * self.diffusion_coefficient *\
                (fe.dot(fe.grad(u), fe.grad(v)) * fe.dx)

        self.rhs = (\
                self.u_n + self.dt * (1-self.u_n) * self.u_n\
                ) * v * fe.dx

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
    def ode_system(self,y,t=0):
        dy = y*0
        dy[0] = y[1]
        dy[1] = -self.wave_speed * y[1] - y[0]*(1-y[0])
        return dy

#==================================================================
    def characterize_speed(self):

        N = 200
        C = np.linspace(0.25,3,N+1)
        print('dc=', C[1]-C[0])
        for k,c in enumerate(C):
            self.wave_speed = c
            self.slope_field(k)
            print(k,'is complete')
            print('-------------')

#==================================================================
    def make_potential_movie(self):

        self.create_movie('potential',\
                self.potential_dir,\
                self.potential_movie_dir,\
                20)


#==================================================================
    def make_slope_field_movie(self):

        self.create_movie('slope_field',\
                self.slope_field_dir,\
                self.slope_field_movie_dir,\
                10)


#==================================================================
    def characterize_potential(self):

        left  = -0.8
        right = 1
        y_lim = 0.35
        N = 200
        x_mesh = np.linspace(left,right,N)
        initial_state = [1,-1]
        t_start = -10
        t_final = 18
        t_mesh = np.linspace(t_start, t_final, 300)
        dt = t_mesh[1]-t_mesh[0]
        self.wave_speed = 3
        potential = lambda x: x**2/2 - x**3/3
        eps=1e-2

        sol = ode_solve(self.ode_system, initial_state, t_mesh)
        txt = 'c = ' + '{:0.2f}'.format(self.wave_speed)
        
        for index, z in enumerate(sol[:,0]):
            print('Working on:', index)
            print('----------------')
            zp = potential(z)
            pure_name =  'potential_' + str(index) + '.png'        
            fname = os.path.join(self.potential_dir, pure_name)

            fig = plt.figure()
            ax  = fig.add_subplot(111)
            ax.set_xlabel('z')
            ax.set_ylabel('U(z)')
            ax.plot(x_mesh, potential(x_mesh), 'r-',\
                    linewidth=3, label = 'U(z)')
            txt = 't='+'{:0.2f}'.format(dt*index)
            ax.text(-0.6,0,txt)

            ax.plot(z,zp,'ko',markersize=8)
            ax.set_xlim([left-eps, right+eps])
            ax.set_ylim([0-eps, y_lim+eps])
            ax.legend(loc=0)
            plt.tight_layout()
            fig.savefig(fname, dpi=300)
            plt.close('all')


        #ax.plot(sol[:,0],potential(sol[:,0]),'b+',markersize=8)


        plt.close('all')


#==================================================================
    def slope_field(self, index=0):

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        N   = 10
        left  = -1
        right = 1
        x_mesh = np.linspace(left,right,N)
        y_mesh = np.linspace(left,right,N)
        X,Y = np.meshgrid(x_mesh, y_mesh)
        v = np.vstack((X.ravel(), Y.ravel()))
        Z = self.ode_system(v)
        U = Z[0].reshape(X.shape)
        V = Z[1].reshape(Y.shape)
        norm = np.sqrt(U**2 + V**2)
        U /= norm
        V /= norm
        ax.quiver(X,Y,U,V)

        initial_state = [1,-1]
        t_start = -100
        t_final = 100
        t_mesh = np.linspace(t_start, t_final, 1000)

        sol = ode_solve(self.ode_system, initial_state, t_mesh)
        txt = 'c = ' + '{:0.2f}'.format(self.wave_speed)

        ax.plot(sol[:,0], sol[:,1], 'b-', linewidth=3, label = txt)
        ax.set_xlabel('z')
        ax.set_ylabel("z'")
        ax.legend(loc=1)
        plt.tight_layout()

        pure_name =  'slope_field_' + str(index) + '.png'        
        fname = os.path.join(self.slope_field_dir, pure_name)
        fig.savefig(fname, dpi=300)
        plt.close('all')


#==================================================================
    def run(self):

        self.set_data_dirs()
        self.create_rhs_fun()
        self.create_mesh()
        self.set_function_spaces()
        #self.plot_mesh()
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


