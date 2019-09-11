import numpy as np

class VectorizeQuantileInterpolation:
    def __init__(self, x, f, tau):
        if x.shape[-1] != 1 or len(x.shape) == 1:
            x, f, tau = (np.expand_dims(i, axis=-1) for i in [x, f, tau])
        self.support = x
        self.num_support_pts = x.shape[-2]
        assert f.shape[-2] == self.num_support_pts and tau.shape[-2] == self.num_support_pts
        self.w = self.compute_quadratic_weights(x, f, tau)

    def compute_quadratic_weights(self, x, f, tau):
        x_right, f_right, tau_right = x[1:], f[1:], tau[1:]
        x_left, f_left, tau_left = x[:-1], f[:-1], tau[:-1]
        w = np.zeros((self.num_support_pts + 1, 3))
        ###
        A = np.zeros((self.num_support_pts - 1, 3, 3))
        b = np.zeros((self.num_support_pts - 1, 3))
        A[:, :, -1] = np.ones((3,))
        A[:, 0, 0] = np.square(x_left).squeeze()
        A[:, 1, 0] = np.square(x_right).squeeze()
        A[:, 2, 0] = (np.square(x_left) + np.square(x_right) + x_left * x_right).squeeze() / 3.
        A[:, 0, 1] = x_left.squeeze()
        A[:, 1, 1] = x_right.squeeze()
        A[:, 2, 1] = (x_left + x_right).squeeze() / 2.
        b[:, 0] = f_left.squeeze()
        b[:, 1] = f_right.squeeze()
        b[:, 2] = ((tau_right - tau_left) / (x_right - x_left)).squeeze()
        w[1: -1] = np.linalg.solve(A, b)
        ###
        w[0, 1] = f[0] / tau[0] if tau[0] != 0. else np.nan
        w[-1, 1] = - (f[-1] / (1. - tau[-1])) if tau[-1] != 0. else np.nan
        w[0, 2] = np.log(f[0]) - f[0] * x[0] / tau[0] if tau[0] != 0. else np.nan
        w[-1, 2] = np.log(f[-1]) + (f[-1] * x[-1] / (1. - tau[-1])) if tau[-1] != 0. else np.nan
        return w

    def __call__(self, xnew):
        interv = (self.support < np.tile(xnew, self.support.shape)).sum(axis=0)
        tail_val_mask = np.logical_not(np.logical_and(interv > 0, interv < self.num_support_pts))
        output = np.square(xnew) * self.w[interv, 0] + xnew * self.w[interv, 1] + self.w[interv, 2]
        output = np.where(tail_val_mask, np.exp(output), output)
        output = np.nan_to_num(output)
        return output

    def grads(self, xnew):
        """
        Gradient with respect to input x.
        Do not confuse with gradient wrt support points and values
        """
        interv = (self.support < np.tile(xnew, self.support.shape)).sum(axis=0)
        tail_val_mask = np.logical_not(np.logical_and(interv > 0, interv < self.num_support_pts))
        grads = 2 * xnew * self.w[interv, 0] + self.w[interv, 1]
        output = np.square(xnew) * self.w[interv, 0] + xnew * self.w[interv, 1] + self.w[interv, 2]
        grads[tail_val_mask] = self.w[interv[tail_val_mask], 1] * np.exp(output[tail_val_mask])
        grads = np.nan_to_num(grads)
        return grads
