import numpy as np
import oracles
import copy


def AcceleratedMetaalgorithmSolver(x_0, f, g, H, K, subproblemCallback, stopCallback, fCallback):
    """
    Solves optimization problem using accelerated metaalgorithm
    :param x_0: start point
    :param f: oracle
    :param g: oracle
    :param H: H param for metaalgorithm
    :param K: max iterations
    :param subproblemCallback:
    :param stopCallback:
    :param fCallback:
    :return: point and stats
    """

    class OmegaOracle(oracles.BaseOracle):
        """
        Oracle for omega function
        """

        def __init__(self, f, x):
            self.f_val = f.func(x)
            self.f_grad = f.grad(x)
            self.x = x

        def func(self, y):
            return self.f_val + np.dot(self.f_grad, (y - self.x))

        def grad(self, y):
            return self.f_grad

        def grad_stoh(self, y, i):
            return self.f_grad[i]

        def metrics(self):
            return {}

    class SubproblemOracle(oracles.BaseOracle):
        """
        Oracle for solving subproblem in metaalgorithm
        """

        def __init__(self, omega, g, x_, H):
            self.omega = omega
            self.g = g
            self.x_ = x_.copy()
            self.H = H

            self.f_calls = 0
            self.g_calls = 0

        def func(self, y):
            self.f_calls += 1
            return self.omega.func(y) + self.g.func(y) + (self.H / 2) * np.sum((y - self.x_) ** 2)

        def grad(self, y):
            self.g_calls += 1
            return self.omega.grad(y) + self.g.grad(y) + self.H * (y - self.x_)

        def grad_stoh(self, y, i):
            self.g_calls += 1
            return self.omega.grad_stoh(y, i) + self.g.grad_stoh(y, i) + self.H * (y[i] - self.x_[i])

        def metrics(self):
            return {'func_call': self.f_calls, 'grad_calls': self.g_calls}

    stats = {
        'iters': 0,
        'gs': [],
        'fs': [],
        'in_iters': []
    }

    A = 0
    y = x_0.copy()
    x = x_0.copy()
    for i in range(K):
        lb = 1 / (2 * H)
        a_new = (lb + np.sqrt(lb**2 + 4 * lb * A)) / 2
        A_new = A + a_new
        x_ = (A * y / A_new) + (a_new * x / A_new)

        y_new, in_iters = subproblemCallback(
            x_, SubproblemOracle(OmegaOracle(f, x_), g, x_, H))

        f_grad = f.grad(y_new)
        g_grad = g.grad(y_new)
        x = x - a_new * f_grad - a_new * g_grad
        y = y_new

        stats['iters'] = i + 1
        stats['gs'].append(np.linalg.norm(f_grad + g_grad))
        stats['fs'].append(fCallback(f, g, y))
        stats['in_iters'].append(in_iters)

        A = A_new

        if stopCallback is not None and stopCallback(f_grad + g_grad):
            return y, stats

    return y, stats


def NesterovAcceleratedSolver(x_0, oracle, settings):
    """
    Nesterov's Fast Coordinate Descent method
    Some default paramenters and lines ware taken from https://github.com/dmivilensky/accelerated-taylor-descent/blob/master/ms%20taylor%20contract.ipynb
    """

    n = x_0.shape[0]

    v = x_0.copy()
    x = x_0.copy()

    A = 0
    beta = 1/2

    Li = settings['Li']
    S = settings['S']
    S_sm = S.sum()
    K = settings['K']

    stop_callback = None
    if 'stop_callback' in settings:
        stop_callback = settings['stop_callback']

    # stats = {
    #    'iters': 0,
    #    'xs': []
    # }

    for iter in range(K):
        # from https://github.com/dmivilensky/accelerated-taylor-descent/blob/master/ms%20taylor%20contract.ipynb
        i = int(np.random.choice(np.linspace(0, n - 1, n), p=S / S_sm))
        # from https://github.com/dmivilensky/accelerated-taylor-descent/blob/master/ms%20taylor%20contract.ipynb
        a = np.roots([S_sm**2, -1, -A]).max()
        A = A + a
        alpha = a / A

        y = (1 - alpha) * x + alpha * v

        grad = oracle.grad_stoh(y, i)

        e = np.zeros(n)
        e[i] = 1

        x = y - grad * e / Li[i]
        v = v - (a * S_sm) * grad * e / (Li[i]**beta)

        # stats['xs'].append(x)

        if stop_callback is not None and stop_callback(x):
            return x, iter + 1

    #stats['iters'] = K
    return x, K


class CompositeMaxOracle(oracles.BaseOracle):
    """
    Oracle for maximization problem
    :param y_0: start point
    :param f: oracle with grad
    :param G: oracle(x, y) with grad_x and grad_y
    :param h: oracle with grad
    :param H: H param for metaalgorithm
    :param K: metaalgorithm iterations
    """

    def __init__(self, y_0, f, G, h, H, K, subproblemCallback, stopCallback):
        self.y_0 = y_0.copy()
        self.y_last = self.y_0
        self.f = f
        self.G = G
        self.h = h
        self.H = H
        self.K = K
        self.subproblemCallback = subproblemCallback
        self.stopCallback = stopCallback

        self.f_calls = 0
        self.g_calls = 0
        self.alg_stats = []

    def optimal_y(self, x):
        self.y_last, stats = AcceleratedMetaalgorithmSolver(  # alg searches for min, so we use it for -G_x(y) + h(y)
            self.y_last, self.h, oracles.KOracle(  # use y from prev run
                oracles.FixedXOracle(self.G, x), -1),
            self.H, self.K, self.subproblemCallback,
            self.stopCallback,
            lambda h__, G_y__, y: self.f.func(x) + self.G.func(x, y) - self.h.func(y))
        self.alg_stats.append(stats)

        # self.y_0 = self.y_last  # use y from prev run
        return self.y_last

    def func(self, x):
        self.f_calls += 1
        return self.G.func(x, self.optimal_y(x))

    def grad(self, x):
        self.g_calls += 1
        return self.G.grad_x(x, self.optimal_y(x))

    def grad_stoh(self, x, i):
        self.g_calls += 1
        return self.G.grad_x_stoh(x, self.optimal_y(x), i)

    def metrics(self):
        return {'func_calls': self.f_calls, 'grad_calls': self.g_calls, 'alg': self.alg_stats}


def SolveSaddle(x_0, y_0, f, G, h, out_settings, out_nesterov_settings, in_settings, in_nesterov_settings):
    """
    Solves saddle optimization problem.
    :param x_0: start point
    :param y_0: start point
    :param f: oracle with grad
    :param G: oracle(x, y) with grad_x and grad_y
    :param h: oracle with grad
    :return: min_x [f(x) + max_y (G(x, y) - h(y))], stats
    """

    g = CompositeMaxOracle(
        y_0, f, G, h, in_settings['H'], in_settings['K'],
        lambda y_00, oracle: NesterovAcceleratedSolver(
            y_00, oracle, in_nesterov_settings),
        in_settings['stop_callback'])

    x, stats = AcceleratedMetaalgorithmSolver(
        x_0, f, g, out_settings['H'], out_settings['K'],
        lambda x_00, oracle: NesterovAcceleratedSolver(
            x_00, oracle, out_nesterov_settings),
        out_settings['stop_callback'],
        lambda f__, g__, x: f.func(x) + G.func(x, g.y_last) - h.func(g.y_last))

    y = g.y_last  # copy.deepcopy(g).optimal_y(x)
    #print(f.func(x_0) + G.func(x_0, y_0) - h.func(y_0))
    #print(f.func(x) + G.func(x, y) - h.func(y))
    return x, y, {'out_stats': stats, 'in_stats': g.metrics()['alg'], 'g':
                  {'func': g.metrics()['func_calls'], 'grad': g.metrics()['grad_calls']}}


class CompositeSaddleOracle:
    def __init__(self, y_0, f, G, h, out_settings, out_nesterov_settings, in_settings, in_nesterov_settings):
        self.y_last = y_0
        self.f = f
        self.G = G
        self.h = h

        self.f_calls = 0
        self.grad_calls = 0
        self.alg_stats = []

    def func(self, x):
        self.f_calls += 1
        return self.f.func(x) + self.G.func(x, self.y_last) - self.h.func(self.y_last)

    def grad(self, x):
        self.grad_calls += 1
        return self.f.grad(x) + self.G.grad_x(x, self.y_last)

    def metrics(self, x):
        pass


def SolveSaddleCatalist(x_0, y_0, f, G, h,
                        catalist_settings,
                        out_settings, out_nesterov_settings,
                        in_settings, in_nesterov_settings):
    """
    Solves saddle optimization problem using catalist.
    :param x_0: start point
    :param y_0: start point
    :param f: oracle with grad
    :param G: oracle(x, y) with grad_x and grad_y
    :param h: oracle with grad
    :return: min_x [f(x) + max_y (G(x, y) - h(y))], stats
    """

    F = CompositeSaddleOracle(y_0, f, G, h,
                              out_settings, out_nesterov_settings,
                              in_settings, in_nesterov_settings)

    zero = oracles.ConstantOracle(0)

    inner_stats = []

    def inner_callback(x_00, oracle):
        x, y, stats = SolveSaddle(x_00, F.y_last, f, G, h,
                                  out_settings, out_nesterov_settings,
                                  in_settings, in_nesterov_settings)
        F.y_last = y
        inner_stats.append(stats)
        return x, stats['out_stats']['iters']

    x, stats = AcceleratedMetaalgorithmSolver(
        x_0, zero, F, catalist_settings['H'], catalist_settings['K'],
        inner_callback, catalist_settings['stop_callback'],
        lambda f_, g_, x: f.func(x) + G.func(x, F.y_last) - h.func(F.y_last))

    return x, F.y_last, {'catalist': stats, 'saddle': inner_stats}
