import oracles
import numpy as np
from scipy.stats import ortho_group


class ExperimentData:
    def __init__(self):
        self.f = None
        self.G = None
        self.h = None

        self.L_f = None
        self.L_G = None
        self.L_h = None

        self.mu_x = None
        self.mu_y = None


def generateRandomGMatrix(n, m, low, high, seed):
    np.random.seed(seed)
    return np.matrix(np.random.random(size=(n, m)) * (high - low) + low)


def generateRandomMatrix(n, eigen_min, eigen_max, seed):
    np.random.seed(seed)
    eigen_vals = [eigen_min, eigen_max]
    rand = np.random.random(size=n - 2) * (eigen_max - eigen_min) + eigen_min

    for i in range(n - 2):
        eigen_vals.append(rand[i])

    # https://blogs.sas.com/content/iml/2012/03/30/geneate-a-random-matrix-with-specified-eigenvalues.html
    o = ortho_group.rvs(n)
    d = np.diag(eigen_vals)
    return np.dot(np.dot(o.T, d), o)


def computeLG(matrix):
    n, m = matrix.shape[0], matrix.shape[1]
    #print(n, m)
    z = np.zeros(shape=(n + m, n + m))
    # print(z.shape)
    z[n:, :m] = (matrix / 2).T
    z[:n, m:] = matrix / 2
    w = np.linalg.eigvals(z)
    # for i in range(n + m):
    #    for j in range(n + m):
    #        print(z[i, j], end='\t')
    #    print()
    # print(w)
    return np.max(w)


def quadraticFormFromHess(A):
    res = A.copy()
    res[np.diag_indices_from(res)] /= 2
    return res


def computeMaxMinEigen(matrix):
    w = np.linalg.eigvals(matrix)
    return np.max(w), np.min(w)


def checkEigenValues(matrix, eigen_min, eigen_max):
    EPS = 1e-9
    L, mu = computeMaxMinEigen(matrix)
    assert np.abs(mu - eigen_min) < EPS
    assert np.abs(L - eigen_max) < EPS


def generateRandomFHMatrix(n, eigen_min, eigen_max, seed):
    A = generateRandomMatrix(n, eigen_min, eigen_max, seed)
    checkEigenValues(A, eigen_min, eigen_max)
    return quadraticFormFromHess(A)


def computeLMu(A):
    B = A.copy()
    B[np.diag_indices_from(B)] *= 2
    return computeMaxMinEigen(B)


def calculateQuadraticFormExperimentParams(f_m, G_m, h_m):
    data = ExperimentData()
    data.f = oracles.QuadraticFormOracle(f_m)
    data.G = oracles.MultiplySaddleOracle(G_m)
    data.h = oracles.QuadraticFormOracle(h_m)

    data.L_f, data.mu_x = computeLMu(f_m)
    data.L_h, data.mu_y = computeLMu(h_m)

    data.L_G = computeLG(G_m)
    return data
