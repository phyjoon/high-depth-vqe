import argparse

import jax
import jax.numpy as jnp
from jax.config import config

import expmgr
import qnnops

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser('Ising Model VQE')
parser.add_argument('--n-qubits', type=int, metavar='N', required=True,
                    help='Number of qubits')
parser.add_argument('--n-layers', type=int, metavar='N', required=True,
                    help='Number of alternating layers')
parser.add_argument('--rot-axis', type=str, metavar='R', default='y',
                    choices=['x', 'y', 'z'],
                    help='Direction of rotation gates.')
parser.add_argument('--g', type=float, metavar='M', required=True,
                    help='Transverse magnetic field')
parser.add_argument('--h', type=float, metavar='M', required=True,
                    help='Longitudinal magnetic field')
parser.add_argument('--train-steps', type=int, metavar='N', default=int(1e3),
                    help='Number of training steps. (Default: 1000)')
parser.add_argument('--lr', type=float, metavar='LR', default=0.01,
                    help='Initial value of learning rate. (Default: 0.01)')
parser.add_argument('--log-every', type=int, metavar='N', default=1,
                    help='Logging every N steps. (Default: 1)')
parser.add_argument('--seed', type=int, metavar='N', required=True,
                    help='Random seed. For reproducibility, the value is set explicitly.')
parser.add_argument('--exp-name', type=str, metavar='NAME', default=None,
                    help='Experiment name. If None, the following format will be used as '
                         'the experiment name: Q{n_qubits}L{n_layers}R{rot_axis}BS{block_size} - g{g}h{h} - S{seed} - LR{lr}')
parser.add_argument('--optimizer-name', type=str, metavar='NAME', default='adam',
                    help='Optimizer name. Supports: adam, nesterov, sgd (Default: adam)')
parser.add_argument('--optimizer-args', type=str, metavar='STR', default=None,
                    help='Additional arguments for the chosen optimizer.\n'
                         'For instance, --optimizer-name=nesterov --optimizer-args="mass:0.1"'
                         ' or --optimizer-name=adam --optimizer-args="eps:1e-8,b1:0.9,b2:0.999"'
                         ' (Default: None)')
parser.add_argument('--scheduler-name', type=str, metavar='NAME', default='constant',
                    help=f'Scheduler name. Supports: {qnnops.supported_schedulers()} '
                         f'(Default: constant)')
parser.add_argument('--checkpoint-path', type=str, metavar='PATH', default=None,
                    help='A checkpoint file path to resume')
parser.add_argument('--jax-enable-x64', action='store_true',
                    help='Enable jax x64 option.')
parser.add_argument('--no-jit', dest='use_jit', action='store_false',
                    help='Disable jit option to loss function.')
parser.add_argument('--no-time-tag', dest='time_tag', action='store_false',
                    help='Omit the time tag from experiment name.')
parser.add_argument('--quiet', action='store_true',
                    help='Quite mode (No training logs)')
parser.add_argument('--use-jacfwd', dest='use_jacfwd', action='store_true',
                    help='Enable the forward mode gradient computation (jacfwd).')
parser.add_argument('--version', type=int, default=1, choices=[1, 2],
                    help='qnnops version (Default: 1)')
args = parser.parse_args()


seed = args.seed
n_qubits, n_layers, rot_axis = args.n_qubits, args.n_layers, args.rot_axis
block_size = n_qubits
g, h = args.g, args.h
if not args.exp_name:
    args.exp_name = f'Q{n_qubits}L{n_layers}R{rot_axis}BS{block_size} - g{g}h{h} - S{seed} - LR{args.lr}'
expmgr.init(project='IsingModel', name=args.exp_name, config=args)

# Construct the hamiltonian matrix of Ising model.
ham_matrix = qnnops.ising_hamiltonian(n_qubits=n_qubits, g=g, h=h)
expmgr.save_array('hamiltonian_matrix.npy', ham_matrix, upload_to_wandb=False)
bandwidth = qnnops.bandwidth(ham_matrix)
expmgr.log_array(bandwidth=bandwidth)

eigval, eigvec = jnp.linalg.eigh(ham_matrix)
eigvec = eigvec.T  # Transpose such that eigvec[i] is an eigenvector, rather than eigenftn[:, i]
ground_state = eigvec[0]
next_to_ground_state = eigvec[1]
next_to_next_to_ground_state = eigvec[2]

print("The lowest eigenvalues (energy) and corresponding eigenvectors (state)")
for i in range(min(5, len(eigval))):
    print(f'| {i}-th state energy={eigval[i]:.4f}')
    print(f'| {i}-th state vector={eigvec[i]}')
expmgr.log_array(
    eigenvalues=eigval,
    ground_state=ground_state,
    next_to_ground_state=next_to_ground_state,
)


def circuit(params):
    return qnnops.alternating_layer_ansatz(params, n_qubits, block_size, n_layers, rot_axis)


def loss(params):
    ansatz_state = circuit(params)
    return qnnops.energy(ham_matrix, ansatz_state) - eigval[0]


def monitor(params, **kwargs):  # use kwargs for the flexibility.
    ansatz_state = circuit(params)
    fidelity_with_ground_state = qnnops.fidelity(ansatz_state, ground_state)
    fidelity_with_next_to_ground = qnnops.fidelity(ansatz_state, next_to_ground_state)
    fidelity_with_next_to_next_to_ground = qnnops.fidelity(ansatz_state, next_to_next_to_ground_state)
    return {
        'fidelity/ground': fidelity_with_ground_state.item(),
        'fidelity/next_to_ground': fidelity_with_next_to_ground.item(),
        'fidelity/next_to_next_to_ground': fidelity_with_next_to_next_to_ground.item()
    }


rng = jax.random.PRNGKey(seed)
_, init_params = qnnops.initialize_circuit_params(rng, n_qubits, n_layers)
trained_params, _ = qnnops.train_loop(
    loss, init_params, args.train_steps, args.lr,
    optimizer_name=args.optimizer_name, optimizer_args=args.optimizer_args,
    scheduler_name=args.scheduler_name,
    monitor=monitor,
    checkpoint_path=args.checkpoint_path,
    use_jit=args.use_jit,
    use_jacfwd=args.use_jacfwd,
)

optimized_state = circuit(trained_params)
expmgr.log_array(optimized_state=optimized_state)
expmgr.save_array('optimized_state.npy', optimized_state)

expmgr.save_config(args)

