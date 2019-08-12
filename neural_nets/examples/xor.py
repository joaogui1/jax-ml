from __future__ import print_function

import itertools

import jax
from jax import jit
import jax.numpy as np
from jax.experimental import stax, optimizers
from jax.experimental.stax import Dense, elementwise

import numpy as onp

import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from activations import sigmoid
from losses import create_loss, crossentropy as cse


Tanh = elementwise(np.tanh)
Sigmoid = elementwise(sigmoid)

init_random_params, net = stax.serial(
    Dense(3), Tanh,
    Dense(1), Sigmoid)

loss = create_loss(net, cse)

def test_all_inputs(inputs, params):
    """Tests all possible xor inputs and outputs"""
    predictions = [int(net(params, inp) > 0.5) for inp in inputs]
    for inp, out in zip(inputs, predictions):
        print(inp, '->', out)
    return (predictions == [onp.bitwise_xor(*inp) for inp in inputs])


loss_grad = jax.jit(jax.grad(loss))
# cse_grad = jax.grad(cse_loss)

if __name__ == "__main__":

    rng = jax.random.PRNGKey(0)

    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    opt_init, opt_update, get_params = optimizers.sgd(0.5)
    _, init_params = init_random_params(rng, (-1, 2))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, loss_grad(params, batch), opt_state)
    print("\nStarting training...")

    for n in itertools.count():
        # Grab a single random input
        x = inputs[onp.random.choice(inputs.shape[0])]
        y = onp.bitwise_xor(x[0], x[1])
        batch = (x, y)

        opt_state = update(next(itercount), opt_state, batch)
      
        params = get_params(opt_state)
        # Every 100 iterations, check whether we've solved XOR
        if not n % 100:
            print('Iteration {}'.format(n))
            if test_all_inputs(inputs, get_params(opt_state)):
                break
