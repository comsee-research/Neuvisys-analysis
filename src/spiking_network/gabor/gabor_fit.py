import numpy as np
from scipy.optimize import curve_fit


def gabor_fit_one_patch(basis):
    num_basis = basis.shape[1]
    xi, yi = np.meshgrid(np.arange(10), np.arange(10))
    true_basis = np.zeros((100, num_basis))
    est_basis = np.zeros((100, num_basis))

    mu_table = np.zeros((2, num_basis))
    lambda_table = np.zeros(num_basis)
    sigma_table = np.zeros((2, num_basis))
    theta_table = np.zeros(num_basis)
    phase_table = np.zeros(num_basis)
    error_table = np.zeros(num_basis)

    for i in range(num_basis):
        print('Iteration:', str(i))

        b1 = basis[0:100, i]
        z = np.reshape(b1[0:100], (10, 10))
        x, g, sse = auto_gabor_surf_one_patch(xi, yi, z)

        est_basis[:, i] = g.flatten()
        true_basis[:, i] = b1
        mu_table[:, i] = [x[0], x[1]]
        lambda_table[i] = x[2]
        sigma_table[:, i] = [x[3], x[4]]
        theta_table[i] = x[5]
        phase_table[i] = x[6]
        error_table[i] = sse
    return est_basis, true_basis, mu_table, lambda_table, sigma_table, theta_table, phase_table, error_table


def auto_gabor_surf_one_patch(xi, yi, z):
    theta = np.pi / 2
    lamb = 5
    sigmax = 5
    sigmay = 5
    phase = 0
    ps = np.zeros(9)
    error = 100

    for muy in range(0, 10, 5):
        for mux in range(0, 10, 5):
            p0 = [mux, muy, lamb, sigmax, sigmay, theta, phase, 1, 0]
            long_zi = z.flatten()
            lb = [-np.inf, -np.inf, 2, 0, 0, 0, 0, 0, -np.inf]
            ub = [np.inf, np.inf, 20, 50, 50, np.pi, 2 * np.pi, np.inf, np.inf]
            try:
                popt, pcov = curve_fit(eval_gabor_one_patch, [xi.flatten(), yi.flatten()], long_zi, p0, bounds=(lb, ub))
                perr = np.sqrt(np.diag(pcov))
                if np.sum(perr) < error:
                    error = np.sum(perr)
                    ps = popt
            except RuntimeError:
                pass


    # H = np.array([[np.cos(ps[5]), -np.sin(ps[5])],
    #               [np.sin(ps[5]), np.cos(ps[5])]])
    # K = H * np.diag([ps[3] ** 2, ps[4] ** 2]) * H
    sse = error
    g = eval_gabor_sep(ps, xi, yi)
    return ps, g, sse


def eval_gabor_one_patch(xdata, mux, muy, lamb, sigmax, sigmay, theta, phase, a, b):
    ps = [mux, muy, lamb, sigmax, sigmay, theta, phase, a, b]
    xi = xdata[0, :]
    yi = xdata[1, :]
    z = eval_gabor_sep(ps, xi, yi)

    return z
    # J = np.zeros((xdata.shape[0], len(ps)))
    # for i in range(len(ps)):
    #     delta = np.zeros(len(ps))
    #     delta[i] = 1e-5
    #     est_z = eval_gabor_sep(ps + delta, xi, yi)
    #     J[:, i] = (est_z[:, i] - z[:, i]) / 1e-5
    #
    # return z, J


def eval_gabor_sep(ps, xi, yi):
    x0 = ps[0]
    y0 = ps[1]
    lamb = ps[2]
    sigmax = ps[3]
    sigmay = ps[4]
    theta = ps[5]
    phase = ps[6]
    a = ps[7]
    b = ps[8]

    xip = (xi - x0) * np.cos(theta) + (yi - y0) * np.sin(theta)
    yip = -(xi - x0) * np.sin(theta) + (yi - y0) * np.cos(theta)

    g_in = np.exp(-xip ** 2 / 2 / sigmax ** 2 - yip ** 2 / 2 / sigmay ** 2) * np.cos(2 * np.pi * xip / lamb + phase)
    return a * g_in + b
