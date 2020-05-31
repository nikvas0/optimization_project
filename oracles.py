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

    def grad_y_stoh(self, x, y, i):
        """
        Computes the grad[i] at point x, y.
        :param x, y: point for computation
        :return: gradient
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def grad_x_stoh(self, x, y, i):
        """
        Computes the grad[i] at point x, y.
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


# oracles for experiments

class MultiplySaddleOracle(BaseSaddleOracle):
    def __init__(self, A):
        self.A = np.array(A)
        self.stat = {'f_calls': 0, 'g_calls_x': 0, 'g_calls_y': 0}

    def func(self, x, y):
        self.stat['f_calls'] += 1
        return np.dot(np.dot(self.A, y), x)

    def grad_x(self, x, y):
        self.stat['g_calls_x'] += 1
        return np.dot(self.A, y)

    def grad_x_stoh(self, x, y, i):
        #self.stat['g_calls_y'] += 1
        return self.grad_x(x, y)[i]

    def grad_y(self, x, y):
        self.stat['g_calls_y'] += 1
        #print(self.A, x, y)
        #print(self.A.shape, x.shape, y.shape)
        #print(type(self.A), type(x), type(y))
        #print('sd', np.dot(x, self.A), np.dot(x, self.A).shape)
        return np.dot(x, self.A)

    def grad_y_stoh(self, x, y, i):
        #self.stat['g_calls_y'] += 1
        return self.grad_y(x, y)[i]

    def metrics(self):
        return self.stat


class QuadraticFormOracle(BaseOracle):
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
        #self.stat['g_calls'] += 1
        return self.grad(x)[i]

    def metrics(self):
        return self.stat
