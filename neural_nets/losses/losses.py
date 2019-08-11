import jax.numpy as np

def mse(y, y_pred): return (np.linalg.norm(y - y_pred)**2) / (2. * y.shape[0])
def crossentropy(y, y_pred): return -np.sum(y * (y_pred + np.finfo(float).eps))
