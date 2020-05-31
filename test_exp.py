import experiment


def test_eigenvals():
    m = experiment.generateRandomMatrix(100, 1, 20, 123)
    experiment.checkEigenValues(m, 1, 20)
    print('test eigenvals: ok')


test_eigenvals()
