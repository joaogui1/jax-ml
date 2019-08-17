from jax import random
import jax.numpy as np

def _get_fans(shape):
    receptive_field = np.prod(shape[:-2])
    if len(shape) >= 2:
        fan_in, fan_out = shape[-2], shape[-1]
    elif len(shape) == 1:
        fan_in, fan_out = shape[0]
    else:
        fan_in, fan_out = 1.
    fan_in *= receptive_field
    fan_out *= receptive_field
    return fan_in, fan_out

def zeros(key, shape, dtype=np.float32): np.zeros(shape, dtype)
def ones(key, shape, dtype=np.float32): np.ones(shape, dtype)

def uniform(lim=1.):
    def init(key, shape, dtype=np.float32):
        return random.uniform(key, shape, dtype, minval=-lim, maxval=lim)
    return init

def normal(stddev=1.):
    def init(key, shape, dtype=np.float32):
        return random.normal(key, shape, dtype) * stddev
    return init


def variance_scaling(scale, mode, distribution):
    def init(key, shape, dtype=np.float32):
        fan_in, fan_out = _get_fans(shape)
        if scale <= 0.:
            raise ValueError(f"scale must be positive float, {scale} given")
        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError(f"Invalid mode argument: {mode}, must be either fan_in, fan_out or fan_avg")
        if mode == "fan_in":
            scale /= fan_in
        elif mode == "fan_out":
            scale /= fan_out
        elif mode == "fan_avg":
            scale /= (fan_in + fan_out) / 2
        if distribution == "truncated_normal":
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(scale) / .87962566103423978
            return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
        elif distribution == "normal":
            return random.normal(key, shape, dtype) * np.sqrt(scale)
        elif distribution == "uniform":
            lim = np.sqrt(3. * scale)
            return random.uniform(key, shape, dtype, minval=-lim, maxval=lim)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

glorot_uniform = variance_scaling(1.0, "fan_avg", "uniform")
glorot_normal = variance_scaling(1.0, "fan_avg", "truncated_normal")
lecun_uniform = variance_scaling(1.0, "fan_in", "uniform")
lecun_normal = variance_scaling(1.0, "fan_in", "truncated_normal")
kaiming_uniform = he_uniform = variance_scaling(2.0, "fan_in", "uniform")
kaiming_normal = he_normal = variance_scaling(2.0, "fan_in", "truncated_normal")
