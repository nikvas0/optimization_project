import oracles
import numpy as np

EPS = 1e-9


def test_quad():
    o = oracles.QuadraticFormOracle(np.array([[1, -1],
                                              [-1, 1]]))
    print(o.func(np.array([2, 3])))
    assert np.abs(o.func(np.array([2, 3])) - (4 + 9 - 2 * 2 * 3)) < EPS
    print(o.grad(np.array([2, 3])))
    assert np.abs(o.grad(np.array([2, 3])) - np.array([-2, 2])).sum() < EPS
    print(o.grad_stoh(np.array([2, 3]), 1))
    assert np.abs(o.grad_stoh(np.array([2, 3]), 1) - 2) < EPS
    print('quadratic: ok')


def test_saddle():
    o = oracles.MultiplySaddleOracle(np.array([[1, -1],
                                               [0, 2]]))
    print(o.func(np.array([1, 1]), np.array([2, 3])))
    assert np.abs(o.func(np.array([1, 1]), np.array(
        [2, 3])) - 5) < EPS

    print('g_x', o.grad_x(np.array([1, 1]), np.array([2, 3])))
    assert np.abs(o.grad_x(np.array([1, 1]), np.array(
        [2, 3])) - np.array([-1, 6])).sum() < EPS
    print('g_y', o.grad_y(np.array([1, 1]), np.array([2, 3])))
    assert np.abs(o.grad_y(np.array([1, 1]), np.array(
        [2, 3])) - np.array([1, 1])).sum() < EPS

    print('g_x_stoh', o.grad_x_stoh(np.array([1, 1]), np.array([2, 3]), 1))
    assert np.abs(o.grad_x_stoh(np.array([1, 1]), np.array(
        [2, 3]), 1) - 6) < EPS
    print('g_y_stoh', o.grad_y_stoh(np.array([1, 1]), np.array([2, 3]), 1))
    assert np.abs(o.grad_y_stoh(np.array([1, 1]), np.array(
        [2, 3]), 1) - 1) < EPS

    print('saddle: ok')


test_quad()
test_saddle()
