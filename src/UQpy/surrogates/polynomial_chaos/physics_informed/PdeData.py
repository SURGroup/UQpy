import numpy as np


class PdeData:
    def __init__(self, geometry_xmax, geometry_xmin, der_orders, bc_normals, bc_x, bc_y):
        """
        Class containing information about PDE solved by Physics-informed PCE

        :param geometry_xmax: list of upper bounds of determnistic variables (geometry and time)
        :param geometry_xmin: list of lower bounds of determnistic variables (geometry and time)
        :param der_orders: list of derivative orders of boundary conditions
        :param bc_normals: normals of boundary conditions
        :param bc_x: coordinates of boundary samples
        :param bc_y: prescribed values of boundary conditions in bc_x
        """
        self.xmax = geometry_xmax
        self.xmin = geometry_xmin
        self.der_orders = der_orders
        self.bc_normals = bc_normals
        self.bc_x = bc_x
        self.bc_y = bc_y

        self.nconst = 0
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

        self.nconst = nconst

        if len(coord) > 0:
            coord = np.concatenate(coord)
            value = np.concatenate(value)
            dirichletbc = np.c_[coord, value]

        self.dirichlet = dirichletbc

    def get_bcsamples(self, order):
        """
        Extract boundary conditions of defined order

        :param order: order of extracted boundary conditions
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
