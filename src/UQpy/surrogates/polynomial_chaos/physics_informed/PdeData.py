import numpy as np
from beartype import beartype


class PdeData:
    @beartype
    def __init__(self, upper_bounds: list,
                 lower_bounds: list,
                 derivative_orders: list,
                 boundary_normals: list,
                 boundary_coordinates: list,
                 boundary_values: list):
        """
        Class containing information about PDE solved by Physics-informed PCE

        :param upper_bounds: list of upper bounds of deterministic variables (geometry and time)
        :param lower_bounds: list of lower bounds of deterministic variables (geometry and time)
        :param derivative_orders: list of derivative orders of boundary conditions
        :param boundary_normals: normals of boundary conditions
        :param boundary_coordinates: coordinates of boundary samples
        :param boundary_values: prescribed values of boundary conditions in bc_x
        """
        self.xmax = upper_bounds
        self.xmin = lower_bounds
        self.der_orders = derivative_orders
        self.bc_normals = boundary_normals
        self.bc_x = boundary_coordinates
        self.bc_y = boundary_values

        self.nconstraints = 0
        self.dirichlet = None
        self.extract_dirichlet()

    def extract_dirichlet(self):
        """
        Extract Dirichlet boundary conditions in form [coordinates, prescribed values]

        :return: extracted Dirichlet bc samples in form [coordinates, prescribed values]
        """
        coord = []
        value = []
        dirichletbc = None

        nconst = 0
        for i in range(len(self.der_orders)):

            if self.der_orders[i] == 0:
                coord.append(self.bc_x[i])
                value.append(self.bc_y[i])
            else:
                nconst = nconst + len((self.bc_x[i]))

        self.nconstraints = nconst

        if len(coord) > 0:
            coord = np.concatenate(coord)
            value = np.concatenate(value)
            dirichletbc = np.c_[coord, value]

        self.dirichlet = dirichletbc

    @beartype
    def get_boundary_samples(self, order: int):
        """
        Extract boundary conditions of defined order

        :param order: derivative order of extracted boundary conditions
        :return: extracted bc samples in form [coordinates, prescribed values]
        """
        coord = []
        value = []
        bcsamples = None

        for i in range(len(self.der_orders)):
            if self.der_orders[i] == order:
                coord.append(self.bc_x[i])
                value.append(self.bc_y[i])

        if len(coord) > 0:
            coord = np.concatenate(coord)
            value = np.array(value).reshape(-1)
            bcsamples = np.c_[coord, value]

        return bcsamples
