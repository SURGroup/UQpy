#!/usr/bin/env python

# This code was adapted from http://sfepy.org/doc-devel/mat_optim.html.
"""
Compare various elastic materials frequency.r.time. uniaxial tension/compression test.

Requires Matplotlib.
"""
from __future__ import absolute_import
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import sys
import six
sys.path.append('.')
from sfepy.base.base import output
from sfepy.base.conf import ProblemConf, get_standard_keywords
from sfepy.discrete import Problem
from sfepy.base.plotutils import plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import numpy as np
from functools import partial

def define( K=8.333, mu_nh=3.846, mu_mr=1.923, kappa=1.923, lam=5.769, mu=3.846 ):
    """Define the problem to solve."""
    from sfepy.discrete.fem.meshio import UserMeshIO
    from sfepy.mesh.mesh_generators import gen_block_mesh
    from sfepy.mechanics.matcoefs import stiffness_from_lame

    def mesh_hook(mesh, mode):
        """
        Generate the block mesh.
        """
        if mode == 'read':
            mesh = gen_block_mesh([2, 2, 3], [2, 2, 4], [0, 0, 1.5], name='el3',
                                  verbose=False)
            return mesh

        elif mode == 'write':
            pass

    filename_mesh = UserMeshIO(mesh_hook)

    options = {
        'nls' : 'newton',
        'ls' : 'ls',
        'ts' : 'ts',
        'save_times' : 'all',
    }

    functions = {
        'linear_pressure' : (linear_pressure,),
        'empty' : (lambda ts, coor, mode, region, ig: None,),
    }

    fields = {
        'displacement' : ('real', 3, 'Omega', 1),
    }

    # Coefficients are chosen so that the tangent stiffness is the same for all
    # material for zero strains.
    materials = {
        'solid' : ({
            'K'  : K, # bulk modulus
            'mu_nh' : mu_nh, # shear modulus of neoHookean term
            'mu_mr' : mu_mr, # shear modulus of Mooney-Rivlin term
            'kappa' : kappa, # second modulus of Mooney-Rivlin term
            # elasticity for LE term
            'D' : stiffness_from_lame(dim=3, lam=lam, mu=mu),
        },),
        'load' : 'empty',
    }

    variables = {
        'u' : ('unknown field', 'displacement', 0),
        'v' : ('test field', 'displacement', 'u'),
    }

    regions = {
        'Omega' : 'all',
        'Bottom' : ('vertices in (z < 0.1)', 'facet'),
        'Top' : ('vertices in (z > 2.9)', 'facet'),
    }

    ebcs = {
        'fixb' : ('Bottom', {'u.all' : 0.0}),
        'fixt' : ('Top', {'u.[0,1]' : 0.0}),
    }

    integrals = {
        'i' : 1,
        'isurf' : 2,
    }
    equations = {
        'linear' : """dw_lin_elastic.i.Omega(solid.D, v, u)
                    = dw_surface_ltr.isurf.Top(load.val, v)""",
        'neo-Hookean' : """dw_tl_he_neohook.i.Omega(solid.mu_nh, v, u)
                         + dw_tl_bulk_penalty.i.Omega(solid.K, v, u)
                         = dw_surface_ltr.isurf.Top(load.val, v)""",
        'Mooney-Rivlin' : """dw_tl_he_neohook.i.Omega(solid.mu_mr, v, u)
                           + dw_tl_he_mooney_rivlin.i.Omega(solid.kappa, v, u)
                           + dw_tl_bulk_penalty.i.Omega(solid.K, v, u)
                           = dw_surface_ltr.isurf.Top(load.val, v)""",
    }

    solvers = {
        'ls' : ('ls.scipy_direct', {}),
        'newton' : ('nls.newton', {
            'i_max'      : 5,
            'eps_a'      : 1e-10,
            'eps_r'      : 1.0,
        }),
        'ts' : ('ts.simple', {
            't0'    : 0,
            't1'    : 1,
            'time_interval'    : None,
            'n_step' : 26, # has precedence over time_interval!
            'verbose' : 1,
        }),
    }

    return locals()

##
# Pressure tractions.
def linear_pressure(ts, coor, mode=None, coef=1, **kwargs):
    if mode == 'qp':
        val = np.tile(coef * ts.step, (coor.shape[0], 1, 1))
        return {'val' : val}

def store_top_u(displacements):
    """Function _store() will be called at the end of each loading step. Top
    displacements will be stored into `displacements`."""
    def _store(problem, ts, state):

        top = problem.domain.regions['Top']
        top_u = problem.get_variables()['u'].get_state_in_region(top)
        displacements.append(np.mean(top_u[:,-1]))

    return _store

def solve_branch(problem, branch_function, material_type):

    eq = problem.conf.equations[material_type]
    problem.set_equations({material_type : eq})

    load = problem.get_materials()['load']
    load.set_function(branch_function)

    out = []
    problem.solve(save_results=False, step_hook=store_top_u(out))
    displacements = np.array(out, dtype=np.float64)

    return displacements

helps = {
    'no_plot' : 'do not show plot window',
}

def plot_mesh(pb):
        # plot mesh for macro problem
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
        ax.set_xlim3d(-1.2, 1.2)
        ax.set_ylim3d(-1.2, 1.2)
        ax.set_zlim3d(-0.01, 3.2)
        ax.set_title('3D plot of macro system')
        plt.show(fig2)
        return None

def one_simulation(material_type, define_args, coef_tension=0.25, coef_compression=-0.25, 
    plot_mesh_bool=False, return_load=False):
    
    #parser = ArgumentParser(description=__doc__,
    #                        formatter_class=RawDescriptionHelpFormatter)
    #parser.add_argument('--version', action='version', version='%(prog)s')
    #options = parser.parse_args()
    output.set_output(filename='sfepy_log.txt', quiet=True)

    required, other = get_standard_keywords()
    # Use this file as the input file.
    conf = ProblemConf.from_file(__file__, required, other, 
        define_args=define_args)

    # Create problem instance, but do not set equations.
    problem = Problem.from_conf(conf, init_equations=False)
    if plot_mesh_bool:
        plot_mesh(problem)

    # Solve the problem. Output is ignored, results stored by using the
    # step_hook.
    linear_tension = partial(linear_pressure, coef=coef_tension)
    u_t = solve_branch(problem, linear_tension, material_type)
    linear_compression = partial(linear_pressure, coef=coef_compression)
    u_c = solve_branch(problem, linear_compression, material_type)

    # Get pressure load by calling linear_*() for each time step.
    ts = problem.get_timestepper()
    load_t = np.array([linear_tension(ts, np.array([[0.0]]), 'qp')['val']
                    for aux in ts.iter_from(0)],
                    dtype=np.float64).squeeze()
    load_c = np.array([linear_compression(ts, np.array([[0.0]]), 'qp')['val']
                    for aux in ts.iter_from(0)],
                    dtype=np.float64).squeeze()

    # Join the branches.
    displacements = np.r_[u_c[::-1], u_t]
    load = np.r_[load_c[::-1], load_t]

    if return_load:
        return displacements, load
    else:
        return displacements

def one_simulation_linear(theta, plot_mesh_bool=False, return_load=False):
    material_type = 'linear'
    theta = np.array(theta).reshape((-1, ))
    define_args = {'lam':theta[0], 'mu':theta[1]} # bulk modulus
    return one_simulation(material_type=material_type, plot_mesh_bool=plot_mesh_bool,
        define_args=define_args, return_load=return_load)

def one_simulation_neo_hookean(theta, plot_mesh_bool=False, return_load=False):
    material_type = 'neo-Hookean'
    theta = np.array(theta).reshape((-1, ))
    define_args = {'mu_nh':theta[0]} # bulk modulus
    return one_simulation(material_type=material_type, plot_mesh_bool=plot_mesh_bool,
        define_args=define_args, return_load=return_load)

def one_simulation_mooney_rivlin(theta, plot_mesh_bool=False, return_load=False):
    material_type = 'Mooney-Rivlin'
    theta = np.array(theta).reshape((-1, ))
    define_args = {'mu_mr':theta[0], 'kappa':theta[1]} # bulk modulus
    return one_simulation(material_type=material_type, plot_mesh_bool=plot_mesh_bool,
        define_args=define_args, return_load=return_load)

def one_simulation_linear_v2(theta, plot_mesh_bool=False, return_load=False):
    material_type = 'linear'
    theta = np.array(theta).reshape((-1, ))
    define_args = {'lam':theta[0], 'mu':theta[1]} # bulk modulus
    return one_simulation(material_type=material_type, plot_mesh_bool=plot_mesh_bool,
        define_args=define_args, return_load=return_load, coef_tension=0.15/5)

def one_simulation_neo_hookean_v2(theta, plot_mesh_bool=False, return_load=False):
    material_type = 'neo-Hookean'
    theta = np.array(theta).reshape((-1, ))
    define_args = {'mu_nh':theta[0]} # bulk modulus
    return one_simulation(material_type=material_type, plot_mesh_bool=plot_mesh_bool,
        define_args=define_args, return_load=return_load, coef_tension=0.15/5)

def one_simulation_mooney_rivlin_v2(theta, plot_mesh_bool=False, return_load=False):
    material_type = 'Mooney-Rivlin'
    theta = np.array(theta).reshape((-1, ))
    define_args = {'mu_mr':theta[0], 'kappa':theta[1]} # bulk modulus
    return one_simulation(material_type=material_type, plot_mesh_bool=plot_mesh_bool,
        define_args=define_args, return_load=return_load, coef_tension=0.15/5)

