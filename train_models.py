# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as onp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

def matmul(a,b,c,np=jnp):
    if c is None:
        c = np.zeros(1)

    return np.dot(a,b)+c

seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42 # for luck
onp.random.seed(seed)

sizes = list(map(int,sys.argv[1].split("-")))
dimensions = [tuple([x]) for x in sizes]
neuron_count = sizes
ops = [matmul]*(len(sizes)-1)

# Initialize with a standard gaussian initialization.
A = []
B = []
for i,(op, a,b) in enumerate(zip(ops, sizes, sizes[1:])):
    A.append(onp.random.normal(size=(a,b))/(b**.5))
    B.append(onp.zeros((b,)))

def run(x, A, B):
    """
    Run the neural network forward on the input x using the matrix A,B.
    """
    for i,(op,a,b) in enumerate(zip(ops,A,B)):
        x = op(x,a,b)
        if i < len(sizes)-2:
            x = x*(x>0)
    return x

def loss(params, inputs, targets):
    logits = run(inputs, params[0], params[1])
    res = (targets-logits.flatten())**2
    return jnp.mean(res)

# generate random training data
params = [A,B]

SAMPLES = 20

optimizer = optax.adam(3e-4)

X = onp.random.normal(size=(SAMPLES, sizes[0]))
Y = onp.array(onp.random.normal(size=SAMPLES)>0,dtype=onp.float32)

loss_grad = jax.grad(loss)

@jax.jit
def update(opt_state, params, batch_x, batch_y):
    grads = loss_grad(params, batch_x, batch_y)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_opt_state, new_params

opt_state = optimizer.init(params)

BS = 4

# Train loop.
for i in range(100):
    if i%10 == 0:
        print('loss', loss(params, X, Y))

    for j in range(0,SAMPLES,BS):
        batch_x = X[j:j+BS]
        batch_y = Y[j:j+BS]

        opt_state, params = update(opt_state, params, batch_x, batch_y)

# Save our amazing model.
# Convert jax arrays back to numpy
save_params = onp.array([[onp.array(a) for a in params[0]], [onp.array(b) for b in params[1]]], dtype=object)
onp.save("models/" + str(seed) + "_" + "-".join(map(str,sizes)), save_params)
