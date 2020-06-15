# Проект по непрерывной оптимизации

Запуск ускоренного метаалгоритма (https://arxiv.org/pdf/2004.08691.pdf) для седловых задач.


* experiments_quadratic.ipynb --- **эксперименты c квадратичными формами**
* experiments_exp1.ipynb и experiments_exp2.ipynb --- **эксперименты с LogSumExp** (в первом случае G -- билинейная форма, во втором --- <x, A(y)x>)

* oracles.py --- код оракулов
* optimization.py --- методы оптимизации
* test_algs.py --- тесты работоспособности методов оптимизации
* experiment.py --- код для генерации данных для экспериментов
* test_exp.py --- тесты генерации данных


Требования:
* scipy
* numpy
* matplotlib
