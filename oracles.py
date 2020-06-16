import numpy as np


class BaseOracle:
    """
    Base class for implementation of oracles. (based on https://github.com/arodomanov/cmc-mipt17-opt-course/blob/master/task4/oracles.py)
    """

    def __init__(self):
        pass

    def func(self, x):
        """
        Computes the value of function at point x.
        :param x: point for computation
        :return: function value
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the grad at point x.
        :param x: point for computation
        :return: gradient
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def grad_stoh(self, x, i):
        """
        Computes the grad's component i at point x.
        :param x: point for computation
        :return: gradient[i]
        """
        raise NotImplementedError('Grad stoh oracle is not implemented.')

    def metrics(self):
        """
        Get metrics
        :return: dict with metrics
        """
        raise NotImplementedError('Metrics oracle is not implemented.')


class BaseSaddleOracle:
    """
    Base class for implementation of saddle oracles.
    """

    def __init__(self):
        pass

    def func(self, x, y):
        """
        Computes the value of function at point x, y.
        :param x, y: point for computation
        :return: function value
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad_x(self, x, y):
        """
        Computes the grad by x at point x, y.
        :param x, y: point for computation
        :return: gradient
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def grad_y(self, x, y):
        """
        Computes the grad by y at point x, y.
        :param x, y: point for computation
        :return: gradient
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def grad_y_stoh(self, x, y, i):
        """
        Computes the grad_y[i] at point x, y.
        :param x, y: point for computation
        :return: gradient
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def grad_x_stoh(self, x, y, i):
        """
        Computes the grad_x[i] at point x, y.
        :param x, y: point for computation
        :return: gradient
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def metrics(self):
        """
        Get metrics
        :return: dict with metrics
        """
        raise NotImplementedError('Metrics oracle is not implemented.')


class KOracle(BaseOracle):
    def __init__(self, f, k):
        self.f = f
        self.k = k

    def func(self, x):
        return self.k * self.f.func(x)

    def grad(self, x):
        return self.k * self.f.grad(x)

    def grad_stoh(self, x, i):
        return self.k * self.f.grad_stoh(x, i)

    def metrics(self):
        return self.f.metrics()


class FixedXOracle(BaseOracle):
    """
    Saddle oracle with fixed x coordinate
    """

    def __init__(self, saddle, x):
        self.saddle = saddle
        self.x = x

    def func(self, y):
        return self.saddle.func(self.x, y)

    def grad(self, y):
        return self.saddle.grad_y(self.x, y)

    def grad_stoh(self, y, i):
        return self.saddle.grad_y_stoh(self.x, y, i)


class PowerOracle(BaseOracle):
    def __init__(self, p, k):
        self.p = p
        self.k = k
        self.stat = {'f_calls': 0, 'g_calls': 0}

    def func(self, x):
        self.stat['f_calls'] += 1
        return self.k * np.abs(x ** self.p).sum()

    def grad(self, x):
        self.stat['g_calls'] += 1
        return self.k * self.p * (x ** (self.p - 1))

    def grad_stoh(self, x, i):
        self.stat['g_calls'] += 1
        return self.k * self.p * (x[i] ** (self.p - 1))

    def metrics(self):
        return self.stat


class MultiplyOracle(BaseSaddleOracle):
    """
    Used only for tests
    """

    def __init__(self, k):
        self.k = k
        self.stat = {'f_calls': 0, 'g_calls': 0}

    def func(self, x, y):
        self.stat['f_calls'] += 1
        return self.k * np.dot(x, y)

    def grad_x(self, x, y):
        self.stat['g_calls'] += 1
        return self.k * y

    def grad_x_stoh(self, x, y, i):
        self.stat['g_calls'] += 1
        return self.k * y[i]

    def grad_y(self, x, y):
        self.stat['g_calls'] += 1
        return self.k * x

    def grad_y_stoh(self, x, y, i):
        self.stat['g_calls'] += 1
        return self.k * x[i]

    def metrics(self):
        return self.stat


class ConstantOracle(BaseOracle):
    def __init__(self, C):
        self.C = C
        self.stat = {'f_calls': 0, 'g_calls': 0}

    def func(self, x):
        self.stat['f_calls'] += 1
        return self.C

    def grad(self, x):
        self.stat['g_calls'] += 1
        return 0

    def metrics(self):
        return self.stat


class SumOracle(BaseOracle):
    """
    Oracle for summing provided oracles
    """

    def __init__(self, lst):
        self.lst = lst
        self.stat = {'f_calls': 0, 'g_calls': 0}

    def func(self, x):
        self.stat['f_calls'] += 1
        res = 0
        for o in self.lst:
            res += o.func(x)
        return res

    def grad(self, x):
        self.stat['g_calls'] += 1
        res = 0
        for o in self.lst:
            res += o.grad(x)
        return res

    def grad_stoh(self, x, i):
        self.stat['g_calls'] += 1
        res = 0
        for o in self.lst:
            res += o.grad_stoh(x)
        return res

    def metrics(self):
        return self.stat


# oracles for experiments

class MultiplySaddleOracle(BaseSaddleOracle):
    """
    Oracle for bilinear form
    """

    def __init__(self, A):
        self.A = np.array(A)
        self.stat = {'f_calls': 0, 'g_calls': 0,
                     'g_calls_x': 0, 'g_calls_y': 0}

    def func(self, x, y):
        self.stat['f_calls'] += 1
        return np.dot(x, np.dot(self.A, y))

    def grad_x(self, x, y):
        self.stat['g_calls_x'] += 1
        return np.dot(self.A, y)

    def grad_x_stoh(self, x, y, i):
        self.stat['g_calls_x'] += 1
        return np.dot(self.A[i], y)

    def grad_y(self, x, y):
        self.stat['g_calls_y'] += 1
        return np.dot(x, self.A)

    def grad_y_stoh(self, x, y, i):
        self.stat['g_calls_y'] += 1
        return np.dot(x, self.A[:, i])

    def metrics(self):
        self.stat['g_calls'] = self.stat['g_calls_x'] + self.stat['g_calls_y']
        return self.stat


class MatrixFromYSaddleOracle(BaseSaddleOracle):
    """
    Oracle for function <x, A(y) x>,
        where A(y) = \sum_{i= 1}^{k} M_k \cdot a_i,
            where a_i = B_i \cdot y
    """

    def __init__(self, matrixes, B):
        self.matrixes = matrixes
        self.B = B
        self.stat = {'f_calls': 0, 'g_calls': 0,
                     'g_calls_x': 0, 'g_calls_y': 0}

    def func(self, x, y):
        b = np.dot(self.B, y)
        A = 0
        for i in range(len(self.matrixes)):
            A += b[i] * self.matrixes[i]
        return np.dot(x.T, np.dot(A, x))

    def grad_x(self, x, y):
        self.stat['g_calls_x'] += 1
        b = np.dot(self.B, y)
        A = 0
        for j in range(len(self.matrixes)):
            A += b[j] * self.matrixes[j]
        return np.dot(2 * A, x)

    def grad_x_stoh(self, x, y, i):
        self.stat['g_calls_x'] += 1
        b = np.dot(self.B, y)
        A = 0
        for j in range(len(self.matrixes)):
            A += b[j] * self.matrixes[j][i]
        return np.dot(2 * A, x)

    def grad_y(self, x, y):
        self.stat['g_calls_y'] += 1
        res = 0
        for j in range(len(self.matrixes)):
            res += np.dot(x,
                          np.dot(self.matrixes[j], x)) * self.B[j] * y
        return res

    def grad_y_stoh(self, x, y, i):
        self.stat['g_calls_y'] += 1
        res = 0
        for j in range(len(self.matrixes)):
            res += np.dot(x, np.dot(self.matrixes[j], x)
                          ) * self.B[j][i] * y[i]
        return res

    def metrics(self):
        self.stat['g_calls'] = self.stat['g_calls_x'] + self.stat['g_calls_y']
        return self.stat


class QuadraticFormOracle(BaseOracle):
    """
    Oracle for quadratic form
    """

    def __init__(self, A):
        self.A = np.array(A)
        self.stat = {'f_calls': 0, 'g_calls': 0}

    def func(self, x):
        self.stat['f_calls'] += 1
        return np.dot(np.dot(x, self.A), x)

    def grad(self, x):
        self.stat['g_calls'] += 1
        return 2 * np.dot(self.A, x)

    def grad_stoh(self, x, i):
        self.stat['g_calls'] += 1
        return 2 * np.dot(self.A[i], x)

    def metrics(self):
        return self.stat


class LogSumExpOracle(BaseOracle):
    """
    Oracle for function $\log(\sum_{k=1} ^ p \exp \langle A_k, x\rangle)$
    """

    def __init__(self, A):
        self.A = np.array(A)
        self.stat = {'f_calls': 0, 'g_calls': 0}

    def func(self, x):
        self.stat['f_calls'] += 1
        # https://github.com/dmivilensky/composite-accelerated-method/blob/master/meta-algorithm-vs-ms.ipynb
        t = np.dot(self.A, x)
        u = t.max()
        t -= u
        return u + np.log(np.sum(np.exp(t)))

    def grad(self, x):
        self.stat['g_calls'] += 1

        # https://github.com/dmivilensky/composite-accelerated-method/blob/master/meta-algorithm-vs-ms.ipynb
        s = np.dot(self.A, x)
        b = s.max()
        z = np.exp(s - b)
        return np.dot(self.A.T, z) / np.dot(np.ones(self.A.shape[0]), z)

    def grad_stoh(self, x, i):
        self.stat['g_calls'] += 1

        # https://github.com/dmivilensky/composite-accelerated-method/blob/master/meta-algorithm-vs-ms.ipynb
        s = np.dot(self.A, x)
        b = s.max()
        z = np.exp(s - b)
        return np.dot(self.A.T[i], z) / np.dot(np.ones(self.A.shape[0]), z)

    def metrics(self):
        return self.stat


class NormOracle(BaseOracle):
    """
    UNUSED
    """

    def __init__(self, G):
        self.G = G
        self.stat = {'f_calls': 0, 'g_calls': 0}

    def func(self, x):
        self.stat['f_calls'] += 1
        return np.dot(x, np.dot(self.G, x)) / 2

    def grad(self, x):
        self.stat['g_calls'] += 1
        return np.dot(self.G, x)

    def grad_stoh(self, x):
        self.stat['g_calls'] += 1
        return np.dot(self.G[i], x)

    def metrics(self):
        return self.stat
