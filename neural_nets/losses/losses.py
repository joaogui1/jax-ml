import jax.numpy as np

def mse(y, y_pred): return np.mean(np.linalg.norm(y - y_pred)**2)
def crossentropy(y, y_pred): return -np.sum(y * np.log(y_pred + np.finfo(float).eps))
