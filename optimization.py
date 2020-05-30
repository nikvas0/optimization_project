import numpy as np
import oracles


def AcceleratedMetaalgorithmSolver(x_0, f, g, H, K, subproblemCallback, stopCallback):
    class SubproblemOracle(oracles.BaseOracle):
        def __init__(self, f, g, x_, H):
            self.omega = oracles.OmegaOracle(f, x_)
            self.g = g
            self.x_ = x_.copy()
            self.H = H

            self.f_calls = 0
            self.g_calls = 0

        def func(self, y):
            self.f_calls += 1
            return self.omega.func(y) + self.g.func(y) + (self.H / 2) * ((y - self.x_) ** 2)

        def grad(self, y):
            self.g_calls += 1
            return self.omega.grad(y) + self.g.grad(y) + self.H * (y - self.x_)

        def metrics(self, ):
            return {'func_call': self.f_calls, 'grad_calls': self.g_calls}

    stats = {
        'iters': 0,
        'ys': [x_0],
        'in_stats': []
    }

    A = 0
    y = x_0.copy()
    x = x_0.copy()
    for i in range(K):
        lb = 1 / (2 * H)
        a_new = (lb + np.sqrt(lb**2 + 4 * lb * A)) / 2
        A_new = A + a_new
        x_ = (A * y + a_new * x) / A_new

        y_new, in_stats = subproblemCallback(SubproblemOracle(f, g, x_, H))

        x = x - a_new * f.grad(y_new) - a_new * g.grad(y_new)
        y = y_new

        stats['iters'] = i + 1
        stats['ys'].append(y)
        stats['in_stats'].append(in_stats)

        if stopCallback(y):
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

    stats = {
        'iters': 0,
        'xs': []
    }

    for _ in range(K):
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

        stats['xs'].append(x)

    stats['iters'] = K
    return x, stats


class CompositeMaxOracle(oracles.BaseOracle):
    def __init__(self, y_0, G, h, H, K, subproblemCallback, stopCallback):
        self.y_0 = y_0.copy()
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
        y, stats = AcceleratedMetaalgorithmSolver(
            self.y_0, oracles.FixedXOracle(
                self.G, x), oracles.MinusOracle(self.h),
            self.H, self.K, self.subproblemCallback,
            self.stopCallback)
        self.alg_stats.append(stats)
        return y

    def func(self, x):
        self.f_calls += 1
        return self.G.func(x, self.optimal_y(x))

    def grad(self, x):
        self.g_calls += 1
        return self.G.grad_x(x, self.optimal_y(x))

    def metrics(self):
        return {'func_call': self.f_calls, 'grad_calls': self.g_calls, 'alg': self.alg_stats}


def SolveSaddle(x_0, y_0, f, G, h, out_settings, out_nesterov_settings, in_settings, in_nesterov_settings):
    """
    Solves saddle optimization problem.
    :param f: oracle with grad
    :param G: oracle(x, y) with grad_x and grad_y
    :param h: oracle with grad
    :return: min_x [f(x) + max_y (G(x, y) - h(y))], stats
    """

    g = CompositeMaxOracle(
        y_0, G, h, in_settings['H'], in_settings['K'],
        lambda oracle: NesterovAcceleratedSolver(
            y_0, oracle, in_nesterov_settings),
        in_settings['stop_callback'])

    res, stats = AcceleratedMetaalgorithmSolver(
        x_0, f, g, out_settings['H'], out_settings['K'],
        lambda oracle: NesterovAcceleratedSolver(
            y_0, oracle, out_nesterov_settings),
        out_settings['stop_callback'])
    return res, {'out_stats': stats, 'in_stats': g.metrics()['alg']}


def SolveSaddleCatalist():
    pass
