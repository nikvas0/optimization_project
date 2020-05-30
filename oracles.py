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
        return self.grad(x)[i]

    def metrics(self):
        """
        Get metrics
        :return: dict with metrics
        """
        raise NotImplementedError('Metrics oracle is not implemented.')


class BaseSaddleOracle:
    """
    Base class for implementation of oracles. (based on https://github.com/arodomanov/cmc-mipt17-opt-course/blob/master/task4/oracles.py)
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
        Computes the grad at point x, y.
        :param x, y: point for computation
        :return: gradient
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def grad_y(self, x, y):
        """
        Computes the grad at point x, y.
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


class MinusOracle(BaseOracle):
    def __init__(self, f):
        self.f = f

    def func(self, x):
        return -self.f.func(x)

    def grad(self, x):
        return -self.f.grad(x)

    def metrics(self):
        return self.f.metrics()


class FixedXOracle(BaseOracle):
    def __init__(self, saddle, x):
        self.saddle = saddle
        self.x = x

    def func(self, y):
        return self.saddle.func(self.x, y)

    def grad(self, y):
        return self.saddle.grad_y(self.x, y)


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
    def __init__(self, k):
        self.k = k
        self.stat = {'f_calls': 0, 'g_calls': 0}

    def func(self, x, y):
        self.stat['f_calls'] += 1
        return np.dot(x, y)

    def grad_x(self, x, y):
        self.stat['g_calls'] += 1
        return y

    def grad_y(self, x, y):
        self.stat['g_calls'] += 1
        return x

    def metrics(self):
        return self.stat
