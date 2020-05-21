#!/usr/bin/env python

# This code was adapted from http://sfepy.org/doc-devel/mat_optim.html.

from __future__ import print_function
from __future__ import absolute_import
import sys
sys.path.append('.')

import matplotlib as mlp
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import numpy as np

from sfepy.base.base import Struct, output
from sfepy.base.log import Log
from sfepy import data_dir

class MaterialSimulator(object):

    @staticmethod
    def create_app(filename, is_homog=False, **kwargs):
        from sfepy.base.conf import ProblemConf, get_standard_keywords
        from sfepy.homogenization.homogen_app import HomogenizationApp
        from sfepy.applications import PDESolverApp

        required, other = get_standard_keywords()
        if is_homog:
            required.remove('equations')

        conf = ProblemConf.from_file(filename, required, other,
                                     define_args=kwargs)
        options = Struct(output_filename_trunk=None,
                         save_ebc=False,
                         save_ebc_nodes=False,
                         save_regions=False,
                         save_regions_as_groups=False,
                         save_field_meshes=False,
                         solve_not=False,
                         )
        output.set_output(filename='sfepy_log.txt', quiet=True)

        if is_homog:
            app = HomogenizationApp(conf, options, 'material_opt_micro:')

        else:
            app = PDESolverApp(conf, options, 'material_opt_macro:')

        app.conf.opt_data = {}
        opts = conf.options
        if hasattr(opts, 'parametric_hook'):  # Parametric study.
            parametric_hook = conf.get_function(opts.parametric_hook)
            app.parametrize(parametric_hook)

        return app

    def __init__(self, macro_fn, micro_fn, phis, plot_meshes_bool=False):
        self.macro_app = self.create_app(macro_fn, is_homog=False, is_opt=True)
        self.micro_app = self.create_app(micro_fn, is_homog=True, is_opt=True)
        self.phis = phis
        self.plot_meshes_bool = plot_meshes_bool

    @staticmethod
    def rotate_mat(D, angle):
        s = np.sin(angle)
        c = np.cos(angle)
        s2 = s**2
        c2 = c**2
        sc = s * c
        T = np.array([[c2, 0, s2, 0, 2*sc,0],
                      [0, 1, 0, 0, 0, 0],
                      [s2, 0, c2, 0, -2*sc, 0],
                      [0, 0, 0, c, 0, -s],
                      [-sc, 0, sc, 0, c2 - s2, 0],
                      [0, 0, 0, s, 0, c]])

        return np.dot(np.dot(T, D), T.T)

    def plot_meshes(self):
        # plot mesh for micro problem
        pb = self.micro_app.problem
        coors = pb.domain.mesh.coors
        #print(set(coors[:,2]))
        graph = pb.domain.mesh.get_conn(pb.domain.mesh.descs[0])
        graph_slice = np.zeros((graph.shape[0], 4))
        for j in range(graph.shape[0]):
            graph_slice[j,:] = graph[j,coors[graph[j,:],2] == 0]
        cells_matrix = pb.domain.regions['Ym'].get_cells()
        cells_fibers = pb.domain.regions['Yf'].get_cells()
        fig = plt.figure(figsize = (12, 5))
        ax = fig.add_subplot(121)
        pc = PolyCollection(verts=coors[graph[cells_matrix,0:4],:2], facecolors='white', 
            edgecolors='black')
        ax.add_collection(pc)
        pc = PolyCollection(verts=coors[graph[cells_fibers,0:4],:2], facecolors='gray', 
            edgecolors='black')
        ax.add_collection(pc)
        ax.axis('equal')
        ax.set_title('2D plot of microstructure')
        ax = fig.add_subplot(122, projection='3d')
        for e in range(graph.shape[0]):
            if e in cells_fibers:
                color = 'gray'
            else:
                color = 'white'
            tupleList = coors[graph[e,:],:]
            vertices = [[0, 1, 2, 3], [4, 5, 6, 7], 
                [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
            verts = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] 
                for ix in range(len(vertices))]
            pc3d = Poly3DCollection(verts=verts, facecolors=color, 
                edgecolors='black', linewidths=1, alpha=0.5)
            ax.add_collection3d(pc3d)
        ax.set_title('3D plot of microstructure')
        plt.show(fig)
        
        # plot mesh for macro problem
        pb = self.macro_app.problem
        coors = pb.domain.mesh.coors
        graph = pb.domain.mesh.get_conn(pb.domain.mesh.descs[0])
        fig2 = plt.figure(figsize=(5,6))
        ax = fig2.add_subplot(111, projection='3d')
        for e in range(graph.shape[0]):
            tupleList = coors[graph[e,:],:]
            vertices = [[0, 1, 2, 3], [4, 5, 6, 7], 
                [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
            verts = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] 
                for ix in range(len(vertices))]
            pc3d = Poly3DCollection(verts=verts, facecolors='white', 
                edgecolors='black', linewidths=1, alpha=0.5)
            ax.add_collection3d(pc3d)
        ax.set_xlim3d(-0.03, 0.03)
        ax.set_ylim3d(-0.01, 0.01)
        ax.set_zlim3d(-0.01, 0.1)
        ax.set_title('3D plot of macro system')
        plt.show(fig2)
        return None

    def mat_eval(self, x):
        mic_od = self.micro_app.conf.opt_data
        mac_od = self.macro_app.conf.opt_data

        mic_od['coefs'] = {}
        mic_od['mat_params'] = x_norm2real(x)
        self.micro_app()

        D = mic_od['D_homog']
        comp_k = []
        for phi in self.phis:
            #print('phi = %d' % phi)

            mac_od['D_homog'] = self.rotate_mat(D, np.deg2rad(phi))
            self.macro_app()

            comp_k.append(mac_od['k'])

        # added by Audrey: get a plot of a slice of the mesh
        if self.plot_meshes_bool:
            self.plot_meshes()
        
        return comp_k

def bounds():
    x_L = [120e9, 0.2, 2e9, 0.2]
    x_U = [200e9, 0.45, 8e9, 0.45]
    return x_L, x_U

def x_norm2real(x):
    x_L, x_U = np.array(bounds())
    return x * (x_U - x_L) + x_L

def x_real2norm(x):
    x_L, x_U = np.array(bounds())
    return (x - x_L) / (x_U - x_L)

micro_filename = data_dir + '/examples/homogenization/' + 'homogenization_opt.py'
macro_filename = data_dir + '/examples/homogenization/' + 'linear_elasticity_opt.py'

def one_simulation(x0, plot_meshes_bool=False):
    """
    This function is the main callable here: it takes in as input the parameter vector, 
    here x0=[E_fiber, nu_fiber, E_matrix, nu_matrix], and returns the simulated output 
    (here slope of the force-elongation curve obtained during a tensile test), to be compared
    with the measured data.
    """
    x0 = x0.reshape((-1, ))
    phis = [0, 30, 60, 90]
    #exp_data = zip([0, 30, 60, 90], [1051140., 197330., 101226., 95474.])
    ms = MaterialSimulator(macro_filename, micro_filename,
                           phis,
                           plot_meshes_bool=plot_meshes_bool)
    qoi = ms.mat_eval(x0)
    return qoi

def one_simulation_2params(x0, plot_meshes_bool=False):
    x0 = x0.reshape((-1, ))
    x0 = np.array([x0[0], 0.45, x0[1], 0.])
    phis = [0, 30, 60, 90]
    #exp_data = zip([0, 30, 60, 90], [1051140., 197330., 101226., 95474.])
    ms = MaterialSimulator(macro_filename, micro_filename,
                           phis, plot_meshes_bool=plot_meshes_bool)

    qoi = ms.mat_eval(x0)
    return qoi

def one_simulation_2params_rvs(x0, plot_meshes_bool=False):
    x0 = x0.reshape((-1, ))
    x0 = np.array([x0[0], 0.45, x0[1], 0.])
    phis = [0, 30, 60, 90]
    ms = MaterialSimulator(macro_filename, micro_filename,
                           phis,
                           plot_meshes_bool=plot_meshes_bool)

    qoi = ms.mat_eval(x0)
    qoi = np.tile(np.array(qoi), 100)
    return qoi
