import numpy as np


def load_data(modes, file_name):
    file1 = np.load(file_name)
    nt, nx, ny = 20, file1["nx"], file1["ny"]
    n_samples = file1["n_samples"]
    inputs = file1["inputs"].reshape(n_samples, nx, ny)
    outputs = np.array((file1["outputs"])).reshape(n_samples, nt, nx, ny)

    # n_samples_ood = file2['n_samples']
    # inputs_ood = file2['inputs'].reshape(n_samples_ood, nx, ny)
    # outputs_ood = np.array((file2['outputs'])).reshape(n_samples_ood, nt, nx, ny)

    num_train = 800
    num_test = 200

    s, t = 28, 20

    f_train = inputs[:num_train, :, :]
    u_train = outputs[:num_train, :, :, :]

    f_test = inputs[num_train : num_train + num_test, :, :]
    u_test = outputs[num_train : num_train + num_test, :, :, :]

    # f_ood = inputs_ood[:num_ood]
    # u_ood = outputs_ood[:num_ood]

    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    z = np.linspace(0, 1, t)

    zz, xx, yy = np.meshgrid(z, x, y, indexing="ij")

    xx = np.reshape(xx, (-1, 1))  # flatten
    yy = np.reshape(yy, (-1, 1))  # flatten
    zz = np.reshape(zz, (-1, 1))  # flatten

    X = np.hstack((zz, xx, yy))  # shape=[t*s*s,3]

    f_train_mean = np.mean(f_train, 0)
    f_train_std = np.std(f_train, 0)
    u_train_mean = np.mean(u_train, 0)
    u_train_std = np.std(u_train, 0)

    # OOD data
    # f_ood_mean = np.mean(f_ood, 0)
    # f_ood_std = np.std(f_ood, 0)
    # u_ood_mean = np.mean(u_ood, 0)
    # u_ood_std = np.std(u_ood, 0)

    num_res = t * s * s  # total output dimension

    # Train data
    f_train_mean = np.reshape(f_train_mean, (-1, s, s, 1))
    f_train_std = np.reshape(f_train_std, (-1, s, s, 1))
    u_train_mean = np.reshape(u_train_mean, (-1, s * s * t, 1))
    u_train_std = np.reshape(u_train_std, (-1, s * s * t, 1))

    # OOD data
    # f_ood_mean = np.reshape(f_ood_mean, (-1, s, s, 1))
    # f_ood_std = np.reshape(f_ood_std, (-1, s, s, 1))
    # u_ood_mean = np.reshape(u_ood_mean, (-1, num_res, 1))
    # u_ood_std = np.reshape(u_ood_std, (-1, num_res, 1))

    #  Mean normalization of train data
    F_train = np.reshape(f_train, (-1, s, s, 1))
    F_train = (F_train - f_train_mean) / (f_train_std + 1.0e-9)
    U_train = np.reshape(u_train, (-1, num_res, 1))
    U_train = (U_train - u_train_mean) / (u_train_std + 1.0e-9)

    #  Mean normalization of test data
    F_test = np.reshape(f_test, (-1, s, s, 1))
    F_test = (F_test - f_train_mean) / (f_train_std + 1.0e-9)
    U_test = np.reshape(u_test, (-1, num_res, 1))
    U_test = (U_test - u_train_mean) / (u_train_std + 1.0e-9)

    #  Mean normalization of ood data
    # F_ood = np.reshape(f_ood, (-1, s, s, 1))
    # F_ood = (F_ood - f_ood_mean)/(f_ood_std + 1.0e-9)
    # U_ood = np.reshape(u_ood, (-1, num_res, 1))
    # U_ood = (U_ood - u_ood_mean)/(u_ood_std + 1.0e-9)

    # Train data
    U = np.reshape(U_train, (-1, num_res))
    C_u = 1.0 / (num_train - 1) * np.matmul(U.T, U)
    lam_u, phi_u = np.linalg.eigh(C_u)

    lam_u = np.flip(lam_u)
    phi_u = np.fliplr(phi_u)
    phi_u = phi_u * np.sqrt(num_res)

    u_basis = phi_u[:, :modes]

    return (
        F_train,
        U_train,
        F_test,
        U_test,
        X,
        u_train_mean,
        u_train_std,
        u_basis,
        lam_u,
    )


def rescale(x, u_mean, u_std):
    x = x * (u_std + 1.0e-9) + u_mean
    return x
