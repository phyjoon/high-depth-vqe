import argparse

import jax
import jax.numpy as jnp
import wandb
from jax.config import config
from tqdm import tqdm
from collections import OrderedDict

from math import log2
import expmgr
import qnnops
import gc

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser('Barren Plateau Test for SYK Model')
parser.add_argument('--n-qubits', type=int, metavar='N', required=True,
                    help='Number of qubits')
parser.add_argument('--max-n-layers', type=int, metavar='N', required=True,
                    help='Maximum number of alternating layers to explore.')
parser.add_argument('--rot-axis', type=str, metavar='R', default='y',
                    choices=['x', 'y', 'z'],
                    help='Direction of rotation gates.')
parser.add_argument('--sample-size', type=int, metavar='N', required=True,
                    help='Size of sample set of gradients.')
parser.add_argument('--seed', type=int, metavar='M', required=True,
                    help='Random seed')
parser.add_argument('--seed-SYK', type=int, metavar='N', required=True,
                    help='Random seed for SYK coupling. For reproducibility, the value is set explicitly.')
parser.add_argument('--jax-enable-x64', action='store_true',
                    help='Enable jax x64 option.')
parser.add_argument('--no-jit', dest='use_jit', action='store_false',
                    help='Disable jit option to loss function.')
parser.add_argument('--no-time-tag', dest='time_tag', action='store_false',
                    help='Omit the time tag from experiment name.')
parser.add_argument('--exp-name', type=str, metavar='NAME', default=None,
                    help='Experiment name. If None, the default format will be used.')
parser.add_argument('--version', type=int, default=1, choices=[1, 2],
                    help='qnnops version (Default: 1)')
args = parser.parse_args()

seed, seed_SYK = args.seed, args.seed_SYK
n_qubits, max_n_layers, rot_axis = args.n_qubits, args.max_n_layers, args.rot_axis
block_size = n_qubits
sample_size = args.sample_size

if not args.exp_name:
    args.exp_name = f'SYK4 - Q{n_qubits}R{rot_axis}BS{block_size} -  SYK{seed_SYK} - S{seed} - SN{sample_size}'
expmgr.init(project='SYK4BP', name=args.exp_name, config=args)

# Construct the hamiltonian matrix of Ising model.
ham_matrix = qnnops.SYK_hamiltonian(jax.random.PRNGKey(args.seed_SYK), n_qubits)
expmgr.save_array('hamiltonian_matrix.npy', ham_matrix, upload_to_wandb=True)

rng = jax.random.PRNGKey(seed)  # Set of random seeds for parameter sampling

M = int(log2(max_n_layers))
for i in range(1, M + 1):
    n_layers = 2 ** i
    print(f'{n_qubits} Qubits & {n_layers} Layers ({i}/{M})')

    def loss(_params):
        ansatz_state = qnnops.alternating_layer_ansatz(
            _params, n_qubits, block_size, n_layers, rot_axis)
        return qnnops.energy(ham_matrix, ansatz_state)

    grad_fn = jax.grad(loss)
    if args.use_jit:
        grad_loss_fn = jax.jit(grad_fn)
        
    # Collect the norms of gradients
    params, grads = [], []
    for step in tqdm(range(sample_size)):
        rng, param_rng = jax.random.split(rng)
        _, param = qnnops.initialize_circuit_params(param_rng, n_qubits, n_layers)
        grad = grad_fn(param)
        params.append(param)
        grads.append(grad)

    params = jnp.vstack(params)
    grads = jnp.vstack(grads)
    grad_norms = jnp.linalg.norm(grads, axis=1)
    
    grads_all_mean, grads_all_var = jnp.mean(grads).item(), jnp.var(grads).item()
    grads_single_mean, grads_single_var = jnp.mean(grads[:, 0]).item(), jnp.var(grads[:, 0]).item()
    grads_norm_mean, grads_norm_var = jnp.mean(grad_norms).item(), jnp.var(grad_norms).item()
    
    logging_output = OrderedDict(
        grad_component_all_mean=grads_all_mean,
        grad_component_all_var=grads_all_var,
        grad_component_single_mean=grads_single_mean,
        grad_component_single_var=grads_single_var,
        grad_norm_mean=grads_norm_mean,
        grad_norm_var=grads_norm_var)
    
    expmgr.log(step=n_layers, logging_output=logging_output)

    wandb.log(
        dict(
            grad_component_all=wandb.Histogram(np_histogram=jnp.histogram(grads, bins=64, density=True)),
            grad_component_single=wandb.Histogram(np_histogram=jnp.histogram(grads[:, 0], bins=64, density=True)),
            grad_norm=wandb.Histogram(np_histogram=jnp.histogram(grad_norms, bins=64, density=True))
        ),
        step=n_layers
    )
    
    suffix = f'Q{n_qubits}L{n_layers}R{rot_axis}BS{block_size}_SYK{seed_SYK}'
    expmgr.save_array(f'params_{suffix}.npy', params)
    expmgr.save_array(f'grads_{suffix}.npy', grads)

    del params, grads
    gc.collect()
