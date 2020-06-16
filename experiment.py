import oracles
import numpy as np
from scipy.stats import ortho_group
import optimization


class ExperimentParams:
    def __init__(self):
        self.f = None
        self.G = None
        self.h = None

        self.L_f = None
        self.L_G = None
        self.L_h = None

        self.mu_x = None
        self.mu_y = None

    def constants(self):
        return {
            'L_f': self.L_f,
            'L_G': self.L_G,
            'L_h': self.L_h,
            'mu_x': self.mu_x,
            'mu_y': self.mu_y,
        }


def generateRandomMatrix(n, eigen_min, eigen_max, seed=None):
    """
    Generates random matrix with eigenvalues from [eigen_min, eigen_max]
    """

    if seed is not None:
        np.random.seed(seed)
    eigen_vals = [eigen_min, eigen_max]
    rand = np.random.random(size=n - 2) * (eigen_max - eigen_min) + eigen_min

    for i in range(n - 2):
        eigen_vals.append(rand[i])

    # https://blogs.sas.com/content/iml/2012/03/30/geneate-a-random-matrix-with-specified-eigenvalues.html
    o = ortho_group.rvs(n)
    d = np.diag(eigen_vals)
    return np.dot(np.dot(o.T, d), o)


def generateRandomGBilinearMatrix(n, m, low, high, seed=None):
    """
    Generates random matrix for bilinear operator
    """

    if seed is not None:
        np.random.seed(seed)
    return np.matrix(np.random.random(size=(n, m)) * (high - low) + low)


def generateRandomGMatrixes(n, k, m, eigen_min, eigen_max, seed=None):
    """
    Generates matrixes for MatrixFromYSaddleOracle init
    """

    if seed is not None:
        np.random.seed(seed)

    ms = []
    for i in range(k):
        ms.append(generateRandomMatrix(n, eigen_min, eigen_max))

    B = np.random.random(size=(k, m))
    for i in range(k):
        B[i] /= np.sum(B[i]) * k  # normalize
        assert np.abs(np.sum(B[i]) - 1 / k) < 1e-9

    return ms, B


def computeLG(matrix):
    """
    Computes Lipschitz constant for the given bilinear form
    """

    n, m = matrix.shape[0], matrix.shape[1]
    # print(n, m)
    z = np.zeros(shape=(n + m, n + m))
    # print(z.shape)
    z[n:, :m] = (matrix / 2).T
    z[:n, m:] = matrix / 2
    w = np.abs(np.linalg.eigvals(2 * z))
    # for i in range(n + m):
    #    for j in range(n + m):
    #        print(z[i, j], end='\t')
    #    print()
    # print(w)
    return np.max(w)


def quadraticFormFromHess(A):
    """
    Computes quadratic form from it's Hessian
    """
    return A / 2


def computeMaxMinEigen(matrix):
    """
    Computes min and max eigenvalues
    """
    w = np.abs(np.linalg.eigvals(matrix))
    return np.max(w), np.min(w)


def checkEigenValues(matrix, eigen_min, eigen_max):
    """
    Checks constants L and \mu
    """
    EPS = 1e-9
    L, mu = computeMaxMinEigen(matrix)
    assert np.abs(mu - eigen_min) < EPS
    assert np.abs(L - eigen_max) < EPS


def generateRandomFHMatrix(n, eigen_min, eigen_max, seed=None):
    """
    Generates random quadratic form with L = eigen_max and mu = eigen_min
    """
    A = generateRandomMatrix(n, eigen_min, eigen_max, seed)
    checkEigenValues(A, eigen_min, eigen_max)
    return quadraticFormFromHess(A)


def computeLMu(A):
    """
    Computes L and mu constants for the provided quadratic form
    """
    return computeMaxMinEigen(2 * A)


def calculateQuadraticFormExperimentParams(f_m, G_m, h_m):
    """
    Use provided matrixes to generate experiment with quadratic forms
    """
    params = ExperimentParams()
    params.f = oracles.QuadraticFormOracle(f_m)
    params.G = oracles.MultiplySaddleOracle(G_m)
    params.h = oracles.QuadraticFormOracle(h_m)

    params.L_f, params.mu_x = computeLMu(f_m)
    params.L_h, params.mu_y = computeLMu(h_m)

    params.L_G = computeLG(G_m)
    return params


def calculateExpExperimentParams(dx, dy, Af, Ah, lf, lh, G_m):
    """
    Use provided params to generate experiment with logSumExp + k||x^2|| and bilinear functions
    """

    params = ExperimentParams()

    params.f = oracles.SumOracle([
        oracles.LogSumExpOracle(Af),
        oracles.QuadraticFormOracle((lf / 2) * np.eye(dx))
    ])

    # params.G = oracles.MatrixFromYSaddleOracle(G_m, G_b)
    params.G = oracles.MultiplySaddleOracle(G_m)

    params.h = oracles.SumOracle([
        oracles.LogSumExpOracle(Ah),
        oracles.QuadraticFormOracle((lh / 2) * np.eye(dy))
    ])

    return params


def calculateExpQExperimentParams(dx, dy, Af, Ah, lf, lh, matrixes, B):
    """
    Use provided params to generate experiment with logSumExp + k||x^2|| and <x, A(y) x> functions
    """

    params = ExperimentParams()

    params.f = oracles.SumOracle([
        oracles.LogSumExpOracle(Af),
        oracles.QuadraticFormOracle((lf / 2) * np.eye(dx))
    ])

    params.G = oracles.MatrixFromYSaddleOracle(matrixes, B)

    params.h = oracles.SumOracle([
        oracles.LogSumExpOracle(Ah),
        oracles.QuadraticFormOracle((lh / 2) * np.eye(dy))
    ])

    return params


def generateQuadraticFormExperiment(dx, dy, f_params, G_params, h_params, seed=None):
    """
    Generates experiment for f, h = quadratic form, G = bilinear form
    """

    if seed is not None:
        np.random.seed(seed)

    G_m = generateRandomGBilinearMatrix(
        dx, dy, G_params['min'], G_params['max'])
    f_m = generateRandomFHMatrix(dx, f_params['mu'], f_params['L'])
    h_m = generateRandomFHMatrix(dy, h_params['mu'], h_params['L'])

    exp = calculateQuadraticFormExperimentParams(f_m, G_m, h_m)

    return exp, f_m, G_m, h_m


def generateExpExperiment(dx, dy, A_params, lf, lh, seed=None):
    """
    Generates experiment for
        f, h =\log(\sum_{i = 1}^{p} \exp(\langle A_i, x \rangle)) + \frac{l (\sum_{i = 1}^{n} x_i^2)}{2},
        G = bilinear form
    :param dx, dy: dimension
    :param A_params: params for A matrixes in LogSumExp fuction
    :param lf, lh: l for f and h
    """

    if seed is not None:
        np.random.seed(seed)

    p = A_params['p']
    sparsity = A_params['sparsity']

    # https://github.com/dmivilensky/composite-accelerated-method/blob/master/meta-algorithm-vs-ms.ipynb
    A = np.zeros(shape=(p, dx))
    A[
        np.random.randint(p, size=int(sparsity * p * dx)),
        np.random.randint(dx, size=int(sparsity * p * dx))
    ] = np.random.random(int(sparsity * p * dx)) * 2 - 1

    A2 = np.zeros(shape=(p, dx))
    A2[
        np.random.randint(p, size=int(sparsity * p * dx)),
        np.random.randint(dx, size=int(sparsity * p * dx))
    ] = np.random.random(int(sparsity * p * dx)) * 2 - 1

    G_m = generateRandomGBilinearMatrix(dx, dy, -1, 1)

    exp = calculateExpExperimentParams(dx, dy, A, A2, lf, lh, G_m)
    return exp


def generateExpQExperiment(dx, dy, A_params, lf, lh, G_params, seed=None):
    if seed is not None:
        np.random.seed(seed)

    p = A_params['p']
    sparsity = A_params['sparsity']

    # https://github.com/dmivilensky/composite-accelerated-method/blob/master/meta-algorithm-vs-ms.ipynb
    A = np.zeros(shape=(p, dx))
    A[
        np.random.randint(p, size=int(sparsity * p * dx)),
        np.random.randint(dx, size=int(sparsity * p * dx))
    ] = np.random.random(int(sparsity * p * dx)) * 2 - 1

    A2 = np.zeros(shape=(p, dx))
    A2[
        np.random.randint(p, size=int(sparsity * p * dx)),
        np.random.randint(dx, size=int(sparsity * p * dx))
    ] = np.random.random(int(sparsity * p * dx)) * 2 - 1

    k = G_params['k']
    matrixes, B = generateRandomGMatrixes(
        dx, k, dy, G_params['eigen_min'], G_params['eigen_max'])

    exp = calculateExpQExperimentParams(dx, dy, A, A2, lf, lh, matrixes, B)
    return exp


def runSaddleExperiment(experiment, settings, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x, y, stats = optimization.SolveSaddle(
        settings['x_0'], settings['y_0'],
        experiment.f, experiment.G, experiment.h,
        settings['out'], settings['out_nesterov'],
        settings['in'], settings['in_nesterov'],
        settings.get('notnegative_y', False))
    return x, y, stats


def runSaddleCatalistExperiment(experiment, settings, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x, y, stats = optimization.SolveSaddleCatalist(
        settings['x_0'], settings['y_0'],
        experiment.f, experiment.G, experiment.h,
        settings['catalist'],
        settings['out'], settings['out_nesterov'],
        settings['in'], settings['in_nesterov'],
        settings.get('notnegative_y', False))
    return x, y, stats
