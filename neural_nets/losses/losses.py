import jax.numpy as np

def mean_squared_error(y, y_pred): return np.mean(np.linalg.norm(y - y_pred)**2, axis=-1)
def mean_absolute_error(y, y_pred): return np.mean(np.linalg.norm(y - y_pred, 1), axis=-1)
def hinge(y, y_pred): return np.mean(np.maximum(1. - y * y_pred, 0.), axis=-1)
def crossentropy(y, y_pred): return -np.sum(y * np.log(y_pred + np.finfo(float).eps) + (1. - y) * np.log(1. - y_pred + np.finfo(float).eps))


def create_loss(net, loss):
    def loss_fn(params, inputs, targets):
        preds = net(params, inputs)
        return loss(targets, preds)
    return loss_fn
