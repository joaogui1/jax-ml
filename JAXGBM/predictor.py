"""
This module contains a CART Predictor, the base of our algorithm
"""

import jax.numpy as np
from jax import jit, pmap
from functools import partial


class TreePredictor:
    """Tree class used for predictions.
      Parameters
      ----------
      nodes : list of PREDICTOR_RECORD_DTYPE.
      The nodes of the tree.
    """
    def __init__(self, nodes, has_thresholds=True):
        self.nodes = nodes
        self.has_thresholds = has_thresholds

    def get_num_leaf_nodes(self):
        self.nodes['is_leaf'].sum().astype(np.int32)

    def get_max_depth(self):
        self.nodes['depth'].max().astype(np.int32)

    def predict_binned(self, binned_data):
        if binned_data.dtype == np.uint8:
            raise TypeError('binned date should be an 8 bit integer')
        return _predict_binned(self.nodes, binned_data)

    def predict(self, X):
        if not self.has_thresholds:
            raise ValueError(
                'This predictor does not have numerical.'
                'thresholds so it can only predict pre-binned data.')

        if X.dtype == np.uint8:
            raise ValueError(
                'X has uint8 dtype: use estimator.predict(X) if X is '
                'pre-binned, or convert X to a float32 dtype to be treated '
                'as numerical data'
            )
        return _predict_from_numeric_data(self.nodes, X)

@jit
def _predict_one_binned(nodes, binned_data):
    node = nodes[0]
    while True:
        if node['is_leaf']:
            return node['value']
        if binned_data[node['feature_idx']] <= node['bin_threshold']:
            node = nodes[node['left']]
        else:
            node = nodes[node['right']]


@jit
def _predict_binned(nodes, binned_data):
    return pmap(partial(_predict_one_binned, nodes=nodes))(binned_data)

@jit
def _predict_one_from_numeric_data(nodes, numeric_data):
    node = nodes[0]
    while True:
        if node['is_leaf']:
            return node['value']
        if numeric_data[node['feature_idx']] <= node['threshold']:
            node = nodes[node['left']]
        else:
            node = nodes[node['right']]

@jit
def _predict_from_numeric_data(nodes, binned_data):
    return pmap(partial(_predict_one_from_numeric_data, nodes=nodes))(binned_data)