import numpy as np

class LeftTail:
    def __init__(self, x, f, tau):
        self.a = f / tau
        self.b = np.log(f) - self.a * x

    def __call__(self, x):
        return np.exp(self.a * x + self.b)

    def grad(self, x):
        return self.a * self(x)


class RightTail:
    def __init__(self, x, f, tau):
        self.a = f / (1. - tau)
        self.b = np.log(f) + self.a * x

    def __call__(self, x):
        return np.exp(- self.a * x + self.b)

    def grad(self, x):
        return - self.a * self(x)


class IntervalFunction:
    def __init__(self, x, f, tau):
        x_square = np.square(x)
        x_cube = x * x_square
        f_cond = np.concatenate([
            x_square.reshape(-1, 1), x.reshape(-1, 1), np.ones((2, 1))
        ], axis=1)
        area_cond = np.array([[
            (x_cube[1] - x_cube[0]) / 3,
            (x_square[1] - x_square[0]) / 2,
            (x[1] - x[0])]])

        A = np.concatenate([f_cond, area_cond], axis=0)
        b = np.concatenate([f.reshape(-1, 1), np.array([[tau[1] - tau[0]]])])
        self.weights = np.linalg.solve(A, b)

    def __call__(self, x):
        return self.weights[0] * np.square(x) + self.weights[1] * x + self.weights[2]

    def grads(self, x):
        """
        Gradient with respect to input x.
        Do not confuse with gradient wrt support points and values
        """
        return 2 * self.weights[0] * x + self.weights[1]



