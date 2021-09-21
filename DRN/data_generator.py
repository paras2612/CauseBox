import numpy as np
import scipy.special
import csv
import sys
import os
from scipy.stats import norm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

storage_path = ''


def lindisc(X, t, p):
    ''' Linear MMD '''

    it = np.where(t > 0)
    ic = np.where(t < 1)

    Xc = X[ic]
    Xt = X[it]

    mean_control = np.mean(Xc, axis=0)
    mean_treated = np.mean(Xt, axis=0)

    c = np.square(2 * p - 1) * 0.25
    f = np.sign(p - 0.5)

    mmd = np.sum(np.square(p * mean_treated - (1 - p) * mean_control))
    mmd = f * (p - 0.5) + np.sqrt(c + mmd)

    return mmd


def get_multivariate_normal_params(m, seed=0):
    np.random.seed(seed)

    if dep:
        mu = np.random.normal(size=m) / 10.
        ''' sample random positive semi-definite matrix for cov '''
        temp = np.random.uniform(size=(m, m))
        temp = .5 * (np.transpose(temp) + temp)
        sig = (temp + m * np.eye(m)) / 100.

    else:
        mu = np.zeros(m)
        sig = np.eye(m)

    return mu, sig


def get_latent(m, seed):
    L = np.array((n * [[]]))
    if m != 0:
        mu, sig = get_multivariate_normal_params(m, seed)
        L = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
    return L


def plot(z, pi0_t1, t, y):
    gridspec.GridSpec(3, 1)

    z_min = np.min(z)  # - np.std(z)
    z_max = np.max(z)  # + np.std(z)
    z_grid = np.linspace(z_min, z_max, 100)

    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ind = np.where(t == 0)
    plt.plot(z[ind], np.squeeze(y[ind, 0]), '+', color='r')
    ind = np.where(t == 1)
    plt.plot(z[ind], np.squeeze(y[ind, 1]), '.', color='b')
    plt.legend(['t=0', 't=1'])

    ax = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    ind = np.where(t == 0)
    mu, std = norm.fit(z[ind])
    p = norm.pdf(z_grid, mu, std)
    plt.plot(z_grid, p, color='r', linewidth=2)
    ind = np.where(t == 1)
    mu, std = norm.fit(z[ind])
    p = norm.pdf(z_grid, mu, std)
    plt.plot(z_grid, p, color='b', linewidth=2)

    plt.savefig(data_path + 'info/' + file_name + '.png')
    plt.close()


# exxample: python data_generator.py 8 8 8 1 1. 1. 4
if __name__ == "__main__":
    mA = int(sys.argv[1])  # dimention of \Gamma
    mB = int(sys.argv[2])  # dimention of \Delta
    mC = int(sys.argv[3])  # dimention of \Upsilon
    mD = int(sys.argv[4])  # dimention of noisy feature
    sc = float(sys.argv[5])  # the logistic growth rate or steepness of the curve
    sh = float(sys.argv[6])  # the x-value of the sigmoid's midpoint
    init_seed = int(sys.argv[7])

    dep = 0  # overwright; dep=0 generates harder datasets
    n_trn = 15000  # int(sys.argv[])
    n_tst = 5000  # int(sys.argv[])
    n = n_trn + n_tst

    seed_coef = 10
    max_dim = 8

    which_benchmark = 'Syn_' + '_'.join(str(item) for item in [sc, sh, dep])

    # A = get_latent(mA, seed_coef*init_seed+0)
    # B = get_latent(mB, seed_coef*init_seed+1)
    # C = get_latent(mC, seed_coef*init_seed+2)
    # D = get_latent(mD, seed_coef*init_seed+3)
    # x = np.hstack((A,B,C, D))
    # AB = np.hstack((A,B))
    # BC = np.hstack((B,C))
    temp = get_latent(3 * max_dim + mD, seed_coef * init_seed + 4)

    # bias same
    A = temp[:, 0:mA]
    B = temp[:, mA:mA + mB]
    C = temp[:, mA + mB:mA + mB + mC]

    # # outcome same
    # C = temp[:, 0:mC]
    # B = temp[:, mC:mC+mB]
    # A = temp[:, mC+mB:mC+mB+mA]

    D = temp[:, mA + mB + mC:mA + mB + mC + mD]

    x = np.concatenate([A, B, C, D], axis=1)
    AB = np.concatenate([A, B], axis=1)
    BC = np.concatenate([B, C], axis=1)

    # coefs = np.ones(shape=mA+mB)
    np.random.seed(1 * seed_coef * init_seed)  # <--
    coefs = np.random.normal(size=mA + mB)
    z = np.dot(AB, coefs)
    pi0_t1 = scipy.special.expit(sc * (z + sh))
    t = np.array([])
    for p in pi0_t1:
        t = np.append(t, np.random.binomial(1, p, 1))

    # coefs = np.ones(shape=mB+mC)
    np.random.seed(2 * seed_coef * init_seed)  # <--
    coefs = np.random.normal(size=mB + mC)
    mu_0 = np.dot(BC ** 1, coefs) / (mB + mC)
    coefs = np.random.normal(size=mB + mC)
    mu_1 = np.dot(BC ** 2, coefs) / (mB + mC)

    y = np.zeros((n, 2))
    np.random.seed(3 * seed_coef * init_seed)  # <--
    y[:, 0] = mu_0 + np.random.normal(loc=0., scale=.1, size=n)
    np.random.seed(3 * seed_coef * init_seed)  # <--
    y[:, 1] = mu_1 + np.random.normal(loc=0., scale=.1, size=n)

    yf = np.array([])
    ycf = np.array([])
    for i, t_i in enumerate(t):
        yf = np.append(yf, y[i, int(t_i)])
        ycf = np.append(ycf, y[i, int(1 - t_i)])

    ##################################################################
    data_path = storage_path + '/data/' + which_benchmark
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    which_dataset = '_'.join(str(item) for item in [mA, mB, mC])
    data_path += '/' + which_dataset + '/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        os.mkdir(data_path + 'info/')

    for perm in range(5):
        np.random.seed(4 * seed_coef * init_seed + perm)  # <--
        file_name = str(init_seed) + '_' + str(perm)
        with open(data_path + file_name + '.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            for i in np.random.permutation(n_trn):
                temp = [t[i], yf[i], ycf[i], mu_0[i], mu_1[i]]
                temp.extend(x[i, :])
                csv_writer.writerow(temp)

    file_name = str(init_seed)
    with open(data_path + file_name + '_test' + '.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        for i in range(n_trn, n):
            temp = [t[i], yf[i], ycf[i], mu_0[i], mu_1[i]]
            temp.extend(x[i, :])
            csv_writer.writerow(temp)

    num_pts = 250
    plot(z[:num_pts], pi0_t1[:num_pts], t[:num_pts], y[:num_pts])

    with open(data_path + 'info/specs.csv', 'a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        temp = [np.mean(t), np.min(pi0_t1), np.max(pi0_t1), np.mean(pi0_t1), np.std(pi0_t1)]
        temp.append(lindisc(x, t, np.mean(t)))
        csv_writer.writerow(temp)
