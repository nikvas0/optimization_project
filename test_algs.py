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


test_nesterov(2, 1, np.array([500, 700]))
test_nesterov(2, 10, np.array([500, -700]))

test_metaalg(2, 1, 2, 1, np.array([500, 700]), np.array([-500, 700]))
test_metaalg(2, 10, 2, 3, np.array([500, -700]), np.array([-500, -700]))
