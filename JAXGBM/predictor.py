"""
This module contains a CART Predictor, the base of our algorithm
"""
from jax import jit
import jax.numpy as np


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

    def predict_binned(self, binned_data, out=None):
        if binned_data.dtype == np.uint8:
            raise TypeError('binned date should be an 8 bit integer')
        if out is None:
            out = np.empty(binned_data.shape[0], dtype=np.float32)
        _predict_binned(self.nodes, binned_data, out)


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


#TODO use pmap
@jit
def _predict_binned(nodes, binned_data, out):
    for i in range(binned_data.shape[0]):
        out[i] = _predict_one_binned(nodes, binned_data[i])
