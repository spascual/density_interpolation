import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import pytest

from basic_quantile_pdf import LeftTail, RightTail, IntervalFunction
from vect_interpolation import VectorizeQuantileInterpolation
from tf_interpolation import TFVectorizeQuantileInterpolation


def _basic_implementation(xtest, x, f, tau):
    left = LeftTail(x[0], f[0], tau[0])
    right = RightTail(x[-1], f[-1], tau[-1])
    interval_1 = IntervalFunction(x[0:2].squeeze(), f[0:2].squeeze(), tau[0:2].squeeze())
    interval_2 = IntervalFunction(x[1:3].squeeze(), f[1:3].squeeze(), tau[1:3].squeeze())
    if xtest < x[0]:
        return left(xtest)
    elif xtest < x[1] and xtest > x[0]:
        return interval_1(xtest)
    elif xtest < x[2] and xtest > x[1]:
        return interval_2(xtest)
    else:
        return right(xtest)


def test_gradient_flow():
    """
    Test gradient-flow from distrubution parameter to evaluation of interpolation.
    """
    X = tf.linspace(-10., 10., 100)
    theta = tf.Variable(1.)
    normal = tfp.distributions.Normal(0., theta)
    u = tf.cast([[0.30], [0.75], [0.8]], tf.float32)
    with tf.GradientTape() as tape:
        x = normal.quantile(u)
        f = normal.prob(x)
        interp = TFVectorizeQuantileInterpolation(x, f, u)
        pdf = interp(X)
        grad = tape.gradient(pdf, [theta])
    assert grad[0] is not None


def test_all_interpolations_agree():
    X = tf.linspace(-10., 10., 100)
    theta = tf.Variable(1.)
    normal = tfp.distributions.Normal(0., theta)
    u = tf.cast([[0.30], [0.75], [0.8]], tf.float32)
    x = normal.quantile(u)
    f = normal.prob(x)

    vect_interp = VectorizeQuantileInterpolation(x.numpy(), f.numpy(), u.numpy())
    vect_pdf = vect_interp(X)

    tf_interp = TFVectorizeQuantileInterpolation(x, f, u)
    tf_pdf = tf_interp(X)

    approx_pdf = np.array([_basic_implementation(xi, x.numpy(), f.numpy(), u.numpy()) for xi in X])

    np.testing.assert_almost_equal(tf_pdf.numpy().reshape(-1, 1), approx_pdf)
    np.testing.assert_almost_equal(vect_pdf.reshape(-1, 1), approx_pdf)

