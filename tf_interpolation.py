import tensorflow as tf

class TFVectorizeQuantileInterpolation:
    def __init__(self, x, f, tau):
        if x.shape[-1] != 1 or len(x.shape) == 1:
            x, f, tau = (tf.expand_dims(i, axis=-1) for i in [x, f, tau])
        num_support_pts = x.shape[-2]
        assert f.shape == tau.shape and x.shape == f.shape
        assert f.shape[-2] == num_support_pts and tau.shape[-2] == num_support_pts
        self.support = x
        self.weights = self.compute_quadratic_weights(x, f, tau)

    @tf.function
    def compute_quadratic_weights(self, x, f, tau):
        tau_right_minus_left = tf.expand_dims(tau[1:] - tau[:-1], axis=-1)
        xleft_right = tf.expand_dims(tf.concat([x[:-1], x[1:]], axis=-1), axis=-1)
        x_right_minus_left = tf.expand_dims(x[1:] - x[:-1], axis=-1)
        fleft_right = tf.expand_dims(tf.concat([f[:-1], f[1:]], axis=-1), axis=-1)

        A1 = tf.concat([tf.square(xleft_right), xleft_right, tf.ones_like(xleft_right)], axis=-1)
        A2 = tf.expand_dims(tf.concat([(tf.reduce_sum(tf.square(xleft_right), axis=-2)
                                        + tf.reduce_prod(xleft_right, axis=-2)) / 3,
                                       tf.reduce_sum(xleft_right, axis=-2) / 2,
                                       tf.ones((*xleft_right.shape[:-2], 1))], axis=-1), axis=-2)
        A = tf.concat([A1, A2], axis=-2)
        b = tf.concat([fleft_right, tau_right_minus_left / x_right_minus_left], axis=-2)
        quadratic_coef = tf.squeeze(tf.linalg.solve(A, b), axis=-1)

        left_tail_slope = tf.expand_dims(f[0] / tau[0], axis=-1)
        left_tail_bias = tf.expand_dims(tf.math.log(f[0]) - f[0] * x[0] / tau[0], axis=-1)
        left_tail = tf.concat([tf.zeros_like(left_tail_bias), left_tail_slope, left_tail_bias],
                              axis=-1)

        right_tail_slope = tf.expand_dims(- (f[-1] / (1. - tau[-1])), axis=-1)
        right_tail_bias = tf.expand_dims(tf.math.log(f[-1]) + (f[-1] * x[-1] / (1. - tau[-1])),
                                         axis=-1)
        right_tail = tf.concat([tf.zeros_like(right_tail_bias), right_tail_slope, right_tail_bias],
                               axis=-1)
        return tf.concat([left_tail, quadratic_coef, right_tail], axis=-2)

    @tf.function
    def __call__(self, X: tf.Tensor):
        intervals = tf.reduce_sum(tf.cast(
            self.support < tf.tile(tf.reshape(X, (1, -1)), self.support.shape), tf.int32), axis=-2)
        tail_val_mask = tf.logical_not(
            tf.logical_and(intervals > 0, intervals < self.support.shape[-2]))
        weights = tf.gather(self.weights, intervals, axis=-2)
        output = tf.square(X) * weights[:, 0] + X * weights[:, 1] + weights[:, 2]
        output = tf.where(tail_val_mask, tf.math.exp(output), output)
        output = tf.where(tf.math.is_nan(output), tf.zeros_like(output), output)
        return output
