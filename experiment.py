import oracles
import numpy as np
from scipy.stats import ortho_group


class ExperimentData:
    def __init__(self):
        self.g = None
        self.G = None
        self.h = None

        self.L_g = None
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


def computeLMu(matrix):
    w = np.linalg.eigvals(matrix)
    return np.max(w), np.min(w)


def checkEigenValues(matrix, eigen_min, eigen_max):
    EPS = 1e-9
    L, mu = computeLMu(matrix)
    assert np.abs(mu - eigen_min) < EPS
    assert np.abs(L - eigen_max) < EPS


def generateQuadraticFormExperiment(dx, dy, exp, seed):
    data = ExperimentData()
    return data


def generateEntropyQuadraticExperiment(dx, dy, seed):
    pass
