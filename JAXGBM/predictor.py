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
        self.nodes['is_leaf'].sum().astype('int32')

    def get_max_depth(self):
        self.nodes['depth'].max().astype('int32')