import experiment
import numpy as np

EPS = 1e-9


def test_eigenvals():
    m = experiment.generateRandomMatrix(100, 1, 20, 123)
    experiment.checkEigenValues(m, 1, 20)
    print('test eigenvals: ok')


# def test_gen():
#    G_m = experiment.generateRandomGMatrix(5, 5, -1, 1, 3)
#    print(G_m)
#    print(experiment.computeLG(G_m))


def test_exp():
    G_m = experiment.generateRandomGMatrix(100, 100, -1, 1, 123)
    f_m = experiment.generateRandomFHMatrix(100, 1, 3, 123)
    h_m = experiment.generateRandomFHMatrix(100, 2, 4, 321)

    exp = experiment.calculateQuadraticFormExperimentParams(f_m, G_m, h_m)
    print(exp.mu_x)
    assert np.abs(exp.mu_x - 1) < EPS
    print(exp.L_f)
    assert np.abs(exp.L_f - 3) < EPS
    print(exp.mu_y)
    assert np.abs(exp.mu_y - 2) < EPS
    print(exp.L_h)
    assert np.abs(exp.L_h - 4) < EPS
    print('L_G:', exp.L_G)
    print('test exp: ok')


def test_exp2():
    exp, _, _, _ = experiment.generateQuadraticFormExperiment(
        100, 100,
        {'L': 10, 'mu': 2},
        {'min': -1, 'max': 1},
        {'L': 5, 'mu': 3},
        1111
    )

    print(exp.mu_x)
    assert np.abs(exp.mu_x - 2) < EPS
    print(exp.L_f)
    assert np.abs(exp.L_f - 10) < EPS
    print(exp.mu_y)
    assert np.abs(exp.mu_y - 3) < EPS
    print(exp.L_h)
    assert np.abs(exp.L_h - 5) < EPS
    print('L_G:', exp.L_G)
    print('test exp: ok')


# test_gen()
test_eigenvals()
test_exp()
test_exp2()
