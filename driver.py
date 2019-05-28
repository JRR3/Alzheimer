#==========================================
# Houston Methodist Research Institute
# May 21, 2019
# Supervisor: Vittorio Cristini
# Developer : Javier Ruiz Ramirez
#==========================================

import sys
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

#==================================================================

sys.path.insert(0, './modules')
from timing_module import TimeIt
from brain_slice_module import BrainSection
from image_manipulation_module import ImageManipulation
#from fem_module import FEMSimulation
#from dirichlet_dirichlet_module import DirichletDirichlet as PDE
#from dirichlet_neumann_module import DirichletNeumann as PDE
#from neumann_neumann_module import NeumannNeumann as PDE
from traveling_wave_module import TravelingWave as PDE
#from time_damped_module import TimeDamped as PDE
#from right_dirichlet_neumann_module import RightDirichletNeumann as PDE

#==================================================================

obj = PDE()
#obj.run()
obj.create_movie()
#obj = ImageManipulation()
#obj.get_radial_plot()

#obj.store_boundary_data()
#obj.generate_raw_data_essential_information()
#obj.load_experimental_data_essentials()
#obj.plot_interpolated_slices()
#obj.plot_sphere()

#obj = FEMSimulation()
#obj.plot_traveling_wave()
#obj.plot_smg()
#exit()
#obj.create_coronal_section_mesh(535)
#obj.create_coronal_section_vectors()
#obj.optimize()
#obj.full_brain_mesh()
#obj.plot_coordinate_map()
#obj.run()
#obj.create_movie()



