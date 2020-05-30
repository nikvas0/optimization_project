import oracles
import optimization
import numpy as np

EPS = 1e-9


def test_nesterov(p, k, x_0):
    oracle = oracles.PowerOracle(p, k)
    y, stats = optimization.NesterovAcceleratedSolver(x_0, oracle, {
        'Li': np.array([p * k, p * k]), 'S': np.array([p * k, p * k]), 'K': 200})
    print('nesterov p =', oracle.p, ':', y)
    # print(stats)
    assert np.sum(y ** 2) < EPS
    print('ok')


def test_metaalg(a, ak, b, bk, x_0, y_0):
    f = oracles.PowerOracle(a, ak)
    g = oracles.PowerOracle(b, bk)

    y, stats = optimization.AcceleratedMetaalgorithmSolver(
        x_0, f, g, a * ak, 200,
        lambda oracle: optimization.NesterovAcceleratedSolver(
            y_0, oracle,
            {
                'Li': np.array([3 * b * bk, 3 * b * bk]),
                'S': np.array([3 * b * bk, 3 * b * bk]),
                'K': 120
            }),
        lambda x: False)

    print('metaalg a =', a, ', b =', b, ':', y)
    # print(stats)
    assert np.sum(y ** 2) < EPS
    print('ok')


def test_saddle():
    f = oracles.PowerOracle(2, 1)
    G = oracles.MultiplyOracle(1)
    h = oracles.PowerOracle(2, 1)
    x_0 = np.array([5, 5])
    y_0 = np.array([1, 1])
    def stop_callback(x): return False
    x, stats = optimization.SolveSaddle(x_0, y_0, f, G, h,
                                        {'H': 20, 'K': 50,
                                         'stop_callback': stop_callback},
                                        {'Li': np.array([20, 20]), 'S': np.array(
                                            [20, 20]), 'K': 10},
                                        {'H': 20, 'K': 20,
                                         'stop_callback': stop_callback},
                                        {'Li': np.array([20, 20]), 'S': np.array([20, 20]), 'K': 10})
    print('saddle:', x)
    assert np.sum(x ** 2) < 1
    print('ok')


test_nesterov(2, 1, np.array([500, 700]))
test_nesterov(2, 10, np.array([500, -700]))

test_metaalg(2, 1, 2, 1, np.array([500, 700]), np.array([-500, 700]))
test_metaalg(2, 10, 2, 3, np.array([500, -700]), np.array([-500, -700]))

test_saddle()
