"""
This module contains the BinMapper class.
BinMapper is used for mapping a real-valued dataset into integer-valued bins
with equally-spaced thresholds.
"""

import time
from functools import partial
import numpy as onp
import jax.numpy as np
from jax import random
from jax import jit, pmap
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin


def _find_binning_thresholds(data,
                             max_bins=256,
                             subsample=200000,
                             random_state=None):
    if 2 > max_bins or max_bins > 256:
        raise ValueError(f'max_bins={max_bins} should be no smaller than 2 '
                         f'and no larger than 256.')
    if random_state is None:
        random_state = int(time.time())
    rng = random.PRNGKey(random_state)
    if subsample is not None and data.shape[0] > subsample:
        subset = random.shuffle(rng, np.arange(data.shape[0]))[:subsample]
        data = data[subset]
    dtype = data.dtype
    if dtype.kind != 'f':
        dtype = np.float32

    percentiles = np.linspace(0, 100, num=max_bins + 1)[1:-1]
    binning_thresholds = []
    for f_idx in range(data.shape[1]):
        col_data = np.array(data[:, f_idx], dtype=dtype, order='C')
        distinct_values = onp.unique(col_data)
        if len(distinct_values) <= max_bins:
            midpoints = (distinct_values[:-1] + distinct_values[1:])
            midpoints *= 0.5
        else:
            midpoints = np.percentile(col_data,
                                      percentiles,
                                      interpolation='midpoint').astype(dtype)
        binning_thresholds.append(midpoints)
    return tuple(binning_thresholds)


def _map_to_bins(data, binning_thresholds=None):
    binned = np.zeros_like(data, dtype=np.uint8, order='F')
    binning_thresholds = tuple(
        np.array(bt, dtype=np.float32, order='C') for bt in binning_thresholds)
    for feature_idx in range(data.shape[1]):
        binned[:, feature_idx] = _map_num_col_to_bins(
            data[:, feature_idx], binning_thresholds[feature_idx])
    return binned


def _map_one_col(binning_thresholds, data):
    left, right = 0, binning_thresholds.shape[0]
    while left < right:
        middle = (right + left - 1) // 2
        if data <= binning_thresholds[middle]:
            right = middle
        else:
            left = middle + 1
    return left


@jit
def _map_num_col_to_bins(data, binning_thresholds):
    """Binary search to the find the bin index for each value in data."""
    return pmap(partial(_map_one_col,
                        binning_thresholds=binning_thresholds))(data)

class BinMapper(BaseEstimator, TransformerMixin):
    """Transformer that maps a dataset into integer-valued bins.
    The bins are created in a feature-wise fashion, with equally-spaced
    quantiles.
    Large datasets are subsampled, but the feature-wise quantiles should
    remain stable.
    If the number of unique values for a given feature is less than
    ``max_bins``, then the unique values of this feature are used instead of
    the quantiles.
    Parameters
    ----------
    max_bins : int, optional (default=256)
        The maximum number of bins to use. If for a given feature the number of
        unique values is less than ``max_bins``, then those unique values
        will be used to compute the bin thresholds, instead of the quantiles.
    subsample : int or None, optional (default=1e5)
        If ``n_samples > subsample``, then ``sub_samples`` samples will be
        randomly choosen to compute the quantiles. If ``None``, the whole data
        is used.
    random_state: int or numpy.random.RandomState or None, \
        optional (default=None)
        Pseudo-random number generator to control the random sub-sampling.
        See `scikit-learn glossary
        <https://scikit-learn.org/stable/glossary.html#term-random-state>`_.
    """
    def __init__(self, max_bins=256, subsample=int(1e5), random_state=None):
        self.max_bins = max_bins
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit data X by computing the binning thresholds.
        Parameters
        ----------
        X: array-like
            The data to bin
        Returns
        -------
        self : object
        """
        X = check_array(X)
        self.numerical_thresholds_ = _find_binning_thresholds(
            X, self.max_bins, subsample=self.subsample,
            random_state=self.random_state)

        self.n_bins_per_feature_ = np.array(
            [thresholds.shape[0] + 1
             for thresholds in self.numerical_thresholds_],
            dtype=np.uint32
        )

        return self

    def transform(self, X):
        """Bin data X.
        Parameters
        ----------
        X: array-like
            The data to bin
        Returns
        -------
        X_binned : array-like
            The binned data
        """
        return _map_to_bins(X, binning_thresholds=self.numerical_thresholds_)