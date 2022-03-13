import logging
from abc import ABC

import numpy as np

from UQpy.transformations import *


class TaylorSeries(ABC):

    @staticmethod
    def _derivatives(
        point_u,
        runmodel_object,
        nataf_object,
        order="first",
        point_x=None,
        point_qoi=None,
        df_step=0.01,
    ):
        if point_u is None and point_x is None:
            raise TypeError("UQpy: Either `point_u` or `point_x` must be specified.")

        list_of_samples = list()
        if point_x is not None:
            if order.lower() == "first" or (order.lower() == "second" and point_qoi is None):
                list_of_samples.append(point_x.reshape(1, -1))
        else:
            z_0 = Correlate(point_u.reshape(1, -1), nataf_object.corr_z).samples_z
            nataf_object.run(samples_z=z_0.reshape(1, -1), jacobian=False)
            temp_x_0 = nataf_object.samples_x
            x_0 = temp_x_0
            list_of_samples.append(x_0)

        for ii in range(point_u.shape[0]):
            y_i1_j = point_u.tolist()
            y_i1_j[ii] = y_i1_j[ii] + df_step

            z_i1_j = Correlate(np.array(y_i1_j).reshape(1, -1), nataf_object.corr_z).samples_z
            nataf_object.run(samples_z=z_i1_j.reshape(1, -1), jacobian=False)
            temp_x_i1_j = nataf_object.samples_x
            x_i1_j = temp_x_i1_j
            list_of_samples.append(x_i1_j)

            y_1i_j = point_u.tolist()
            y_1i_j[ii] = y_1i_j[ii] - df_step
            z_1i_j = Correlate(np.array(y_1i_j).reshape(1, -1), nataf_object.corr_z).samples_z
            nataf_object.run(samples_z=z_1i_j.reshape(1, -1), jacobian=False)
            temp_x_1i_j = nataf_object.samples_x
            x_1i_j = temp_x_1i_j
            list_of_samples.append(x_1i_j)

        array_of_samples = np.array(list_of_samples)
        array_of_samples = array_of_samples.reshape((len(array_of_samples), -1))

        runmodel_object.run(samples=array_of_samples, append_samples=False)
        y1 = runmodel_object.qoi_list
        logging.getLogger(__name__).info(
            "samples to evaluate the model: {0}".format(array_of_samples)
            + "model evaluations: {0}".format(runmodel_object.qoi_list))

        if order.lower() == "first":
            gradient = np.zeros(point_u.shape[0])

            for jj in range(point_u.shape[0]):
                qoi_plus = y1[2 * jj + 1]
                qoi_minus = y1[2 * jj + 2]
                gradient[jj] = (qoi_plus - qoi_minus) / (2 * df_step)

            return gradient, y1[0], array_of_samples

        elif order.lower() == "second":
            logging.getLogger(__name__).info("UQpy: Calculating second order derivatives..")
            d2y_dj = np.zeros([point_u.shape[0]])

            if point_qoi is None:
                qoi = [runmodel_object.qoi_list[0]]
                output_list = runmodel_object.qoi_list
            else:
                qoi = [point_qoi]
                output_list = qoi + runmodel_object.qoi_list

            for jj in range(point_u.shape[0]):
                qoi_plus = output_list[2 * jj + 1]
                qoi_minus = output_list[2 * jj + 2]

                d2y_dj[jj] = (qoi_minus - 2 * qoi[0] + qoi_plus) / (df_step ** 2)

            list_of_mixed_points = list()
            import itertools

            range_ = list(range(point_u.shape[0]))
            d2y_dij = np.zeros([int(point_u.shape[0] * (point_u.shape[0] - 1) / 2)])
            count = 0
            for i in itertools.combinations(range_, 2):
                y_i1_j1 = point_u.tolist()
                y_i1_1j = point_u.tolist()
                y_1i_j1 = point_u.tolist()
                y_1i_1j = point_u.tolist()

                y_i1_j1[i[0]] += df_step
                y_i1_j1[i[1]] += df_step

                y_i1_1j[i[0]] += df_step
                y_i1_1j[i[1]] -= df_step

                y_1i_j1[i[0]] -= df_step
                y_1i_j1[i[1]] += df_step

                y_1i_1j[i[0]] -= df_step
                y_1i_1j[i[1]] -= df_step

                z_i1_j1 = Correlate(np.array(y_i1_j1).reshape(1, -1), nataf_object.corr_z).samples_z
                nataf_object.run(samples_z=z_i1_j1.reshape(1, -1), jacobian=False)
                x_i1_j1 = nataf_object.samples_x
                list_of_mixed_points.append(x_i1_j1)

                z_i1_1j = Correlate(np.array(y_i1_1j).reshape(1, -1), nataf_object.corr_z).samples_z
                nataf_object.run(samples_z=z_i1_1j.reshape(1, -1), jacobian=False)
                x_i1_1j = nataf_object.samples_x
                list_of_mixed_points.append(x_i1_1j)

                z_1i_j1 = Correlate(np.array(y_1i_j1).reshape(1, -1), nataf_object.corr_z).samples_z
                nataf_object.run(samples_z=z_1i_j1.reshape(1, -1), jacobian=False)
                x_1i_j1 = nataf_object.samples_x
                list_of_mixed_points.append(x_1i_j1)

                z_1i_1j = Correlate(np.array(y_1i_1j).reshape(1, -1), nataf_object.corr_z).samples_z
                nataf_object.run(samples_z=z_1i_1j.reshape(1, -1), jacobian=False)
                x_1i_1j = nataf_object.samples_x
                list_of_mixed_points.append(x_1i_1j)

                count = count + 1

            array_of_mixed_points = np.array(list_of_mixed_points)
            array_of_mixed_points = array_of_mixed_points.reshape((len(array_of_mixed_points), -1))
            runmodel_object.run(samples=array_of_mixed_points, append_samples=False)

            logging.getLogger(__name__).info(
                "samples for gradient: {0}".format(array_of_mixed_points[1:])
                + "model evaluations for the gradient: {0}".format(runmodel_object.qoi_list[1:]))

            for j in range(count):
                qoi_0 = runmodel_object.qoi_list[4 * j]
                qoi_1 = runmodel_object.qoi_list[4 * j + 1]
                qoi_2 = runmodel_object.qoi_list[4 * j + 2]
                qoi_3 = runmodel_object.qoi_list[4 * j + 3]
                d2y_dij[j] = (qoi_0 + qoi_3 - qoi_1 - qoi_2) / (4 * df_step * df_step)

            hessian = np.diag(d2y_dj)
            import itertools

            range_ = list(range(point_u.shape[0]))
            add_ = 0
            for i in itertools.combinations(range_, 2):
                hessian[i[0], i[1]] = d2y_dij[add_]
                hessian[i[1], i[0]] = hessian[i[0], i[1]]
                add_ += 1

            return hessian
