import gc
import itertools
import pickle
from collections import OrderedDict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
import qutip
from jax.experimental import optimizers

import expmgr
import gate_jax as gates

from itertools import combinations
from math import factorial

jax.config.update("jax_enable_x64", True)

version = expmgr.version()


def create_target_states(n_qubits, n_samples, seed=None):
    """ Create multiple target states with given qubit number.

    Args:
        n_qubits: int, number of qubits
        n_samples: int, number of samples
        seed: int, random seed
    Returns:
        jnp.ndarray, state vectors of shape (n_samples, 2^n_qubits)
    """

    dim = 2 ** n_qubits
    haar_random_states = [
        qutip.rand_ket_haar(N=dim, seed=seed).get_data().toarray().T
        for _ in range(n_samples)]
    return jnp.vstack(haar_random_states)


def initialize_circuit_params(rng, n_qubits, n_layers):
    """ Initialize a state

    Args:
        rng: PRNGKey, random generation key
        n_qubits: int, number of qubits
        n_layers: int, number of layers

    Returns:
        PRNGKey, random generation key
        jnp.ndarray: initial random values in [0, 2pi)
    """

    rng, sub_rng = jax.random.split(rng)
    # TODO(jdk): Do we need to change this as a list of params rather than
    #  flatten vector? Like, [random(n_qubits) for _ in range(n_layers)]
    params = jax.random.uniform(sub_rng, (n_qubits * n_layers,)) * 2 * jnp.pi
    return rng, params


def state_norm(state):
    """ Compute the norm of a state """
    return jnp.real(jnp.sum(state * state.conj()))  # norm must be real.


def block(params, qubits, state, n_qubit, rot_axis='Y'):
    if version == 1:  # version 1 (old) or 2
        return block_v1(params, qubits, state, n_qubit, rot_axis)
    return block_v2(params, qubits, state, n_qubit, rot_axis)


def block_v1(params, qubits, state, n_qubit, rot_axis='Y'):
    rot_axis = rot_axis.upper()
    if rot_axis == 'X':
        rotation_gate = gates.rx
    elif rot_axis == 'Y':
        rotation_gate = gates.ry
    elif rot_axis == 'Z':
        rotation_gate = gates.rz
    else:
        raise ValueError("rot_axis should be either 'X', 'Y', or 'Z'.")

    # Rotation layer
    for qubit, param in zip(qubits, params):
        state = rotation_gate(param, n_qubit, qubit) @ state

    # CZ layer
    entangler_pairs = sorted(
        itertools.combinations(range(len(qubits)), 2),
        key=lambda x: abs(x[0] - x[1]), reverse=False)

    for control, target in entangler_pairs:
        state = gates.cz_gate(n_qubit, control, target) @ state

    return state


def block_v2(params, qubits, state, n_qubit, rot_axis='Y'):
    rot_axis = rot_axis.upper()
    if rot_axis == 'X':
        rotation_gate = gates.rx
    elif rot_axis == 'Y':
        rotation_gate = gates.ry
    elif rot_axis == 'Z':
        rotation_gate = gates.rz
    else:
        raise ValueError("rot_axis should be either 'X', 'Y', or 'Z'.")

    # Rotation layer
    for qubit, param in zip(qubits, params):
        gate = jax.jit(rotation_gate, static_argnums=(1, 2))
        state = gate(param, n_qubit, qubit) @ state

    # CZ layer
    entangler_pairs = sorted(
        itertools.combinations(range(len(qubits)), 2),
        key=lambda x: abs(x[0] - x[1]), reverse=False)

    for control, target in entangler_pairs:
        gate = jax.jit(gates.cz_gate, static_argnums=(0, 1, 2))
        state = gate(n_qubit, control, target) @ state

    return state


def alternating_layer_ansatz(params, n_qubits, block_size, n_layers, rot_axis='Y'):
    # TODO(jdk): Check this function later whether we need to revise for scalability.
    rot_axis = rot_axis.upper()
    assert rot_axis in ('X', 'Y', 'Z')
    assert n_qubits % block_size == 0
    assert len(params) == n_qubits * n_layers

    # Initial state
    state = jnp.array([0] * (2 ** n_qubits - 1) + [1], dtype=jnp.complex64)

    for d in range(n_layers):
        block_indices = jnp.arange(n_qubits)
        if d % 2:
            block_indices = jnp.roll(block_indices, -(block_size // 2))
        block_indices = jnp.reshape(block_indices, (-1, block_size))
        for block_idx in block_indices:
            state = block(params=params[block_idx + d * n_qubits],
                          qubits=block_idx, state=state, n_qubit=n_qubits,
                          rot_axis=rot_axis)

    return state


def get_optimizer(name, optim_args, scheduler):
    name = name.lower()
    if optim_args and isinstance(optim_args, str):
        optim_args = [kv.split(':') for kv in optim_args.split(',')]
        optim_args = {k: float(v) for k, v in optim_args}
    optim_args = optim_args or {}
    if name == 'adam':
        init_fun, update_fun, get_params = optimizers.adam(scheduler, **optim_args)
    elif name == 'nesterov':
        if 'mass' not in optim_args:
            optim_args['mass'] = 0.1
        init_fun, update_fun, get_params = optimizers.nesterov(scheduler, **optim_args)
    else:
        raise ValueError(f'An optimizer {name} is not supported. ')
    print(f'Loaded an optimization {name} - {optim_args}')
    return init_fun, update_fun, get_params


def supported_schedulers():
    return 'constant', 'inverse_time_decay', 'exponential_decay'


def get_scheduler(lr, train_steps, name='constant'):
    name = name.lower()
    if name == 'constant':
        scheduler = optimizers.constant(lr)
    elif name == 'inverse_time_decay':
        decay_steps = int(train_steps // 5)
        scheduler = optimizers.inverse_time_decay(lr, decay_steps, 2)
    elif name == 'exponential_decay':
        decay_steps = int(train_steps // 3)
        scheduler = optimizers.exponential_decay(lr, decay_steps, 0.3)
    else:
        raise ValueError(f'Not supported scheduler {name}.'
                         f'Supported schedulers={supported_schedulers()}')
    print(f'Loaded a scheduler {name} - {scheduler}')
    return scheduler


def train_loop(loss_fn, init_params, train_steps=int(1e4), lr=0.01,
               optimizer_name='adam', optimizer_args=None,
               scheduler_name='constant',
               loss_args=None, early_stopping=False, monitor=None,
               log_every=1, checkpoint_path=None, use_jit=True,
               use_jacfwd=False):
    """ Training loop.

    Args:
        loss_fn: callable, loss function whose first argument must be params.
        init_params: jnp.array, initial trainable parameter values
        train_steps: int, total number of training steps
        lr: float, initial learning rate
        optimizer_name: str, optimizer name to be used.
        optimizer_args: dict, custom arguments for the optimizer.
            If None, default arguments will be used.
        scheduler_name: str, scheduler name.
        loss_args: dict, additional loss arguments if needed.
        early_stopping: bool, whether to early stop if the train loss value
            doesn't decrease further. (Not implemented yet)
        monitor: callable -> dict, monitoring function on training.
        log_every: int, logging every N steps.
        checkpoint_path: str, a checkpoint file path to resume.
        use_jit: bool, whether to use jit compilation.
        use_jacfwd: bool, enable the forward mode jax.jacfwd for gradient
            computation instead the reverse mode jax.grad (jax.jacrev)
            For backward compatibility, this option disables by default.
            But, later it will enable. (Default: False)
    Returns:
        params: jnp.array, optimized parameters
        history: dict, training history.
    """

    assert monitor is None or callable(monitor), 'the monitoring function must be callable.'

    loss_args = loss_args or {}
    train_steps = int(train_steps)  # to guarantee an integer type value.
    scheduler = get_scheduler(lr, train_steps, scheduler_name)
    init_fun, update_fun, get_params = get_optimizer(optimizer_name, optimizer_args, scheduler)
    if checkpoint_path:
        start_step, optimizer_state, history = load_checkpoint(checkpoint_path)
        min_loss = jnp.hstack(history['loss']).min()
    else:
        start_step = 0
        optimizer_state = init_fun(init_params)
        history = {'loss': [], 'grad': [], 'params': []}
        min_loss = float('inf')

    try:
        if use_jacfwd:
            grad_loss_fn = jax.jacfwd(loss_fn)
        else:
            grad_loss_fn = jax.value_and_grad(loss_fn)
        if use_jit:
            grad_loss_fn = jax.jit(grad_loss_fn)
        for step in range(start_step, train_steps):
            params = get_params(optimizer_state)
            if use_jacfwd:
                loss = loss_fn(params, **loss_args)
                grad = grad_loss_fn(params, **loss_args)
            else:
                # for backward compatibility. It will be replaced by
                # jax.grad to make the consistency with jax.jacfwd.
                loss, grad = grad_loss_fn(params, **loss_args)
            optimizer_state = update_fun(step, grad, optimizer_state)
            updated_params = get_params(optimizer_state)

            grad = onp.array(grad)
            params = onp.array(params)
            updated_params = onp.array(updated_params)
            history['loss'].append(loss)
            history['grad'].append(grad)
            history['params'].append(params)
            if loss < min_loss:
                min_loss = loss
                expmgr.save_array('params_best.npy', updated_params)
                save_checkpoint('checkpoint_best.pkl', step, optimizer_state, history)

            if step % log_every == 0:
                grad_norm = jnp.linalg.norm(grad).item()
                logging_output = OrderedDict(loss=loss.item(), lr=scheduler(step), grad_norm=grad_norm)
                if monitor is not None:
                    logging_output.update(monitor(params=params))
                logging_output['min_loss'] = min_loss.item()
                expmgr.log(step, logging_output)
                expmgr.save_array('params_last.npy', updated_params)
                save_checkpoint('checkpoint_last.pkl', step, optimizer_state, history)

            if early_stopping:
                # TODO(jdk): implement early stopping feature.
                pass
            del loss, grad
            gc.collect()

    except Exception as e:
        print(e)
        print('Saving history object...')
        expmgr.save_history('history.npz', history)
        raise e
    else:
        expmgr.save_history('history.npz', history)
    return get_params(optimizer_state), history


def save_checkpoint(filename, step, optimizer_state, history):
    """ Save a training checkpoint """
    pytree = optimizers.unpack_optimizer_state(optimizer_state)
    checkpoint_path = expmgr.get_result_path(filename)
    with checkpoint_path.open('wb') as f:
        pickle.dump(dict(step=step, state=pytree, history=history), f)
    expmgr.safe_wandb_save(checkpoint_path)
    return checkpoint_path


def load_checkpoint(filepath):
    """ Load a training checkpoint """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    with filepath.open('rb') as f:
        checkpoint = pickle.load(f)
    step = checkpoint['step']
    optimizer_state = optimizers.pack_optimizer_state(checkpoint['state'])
    history = checkpoint['history']
    return step + 1, optimizer_state, history


PauliBasis = jnp.array([[[1., 0., ], [0., 1., ]],
                        [[0., 1., ], [1., 0., ]],
                        [[0., -1j, ], [1j, 0., ]],
                        [[1., 0., ], [0., -1., ]]], dtype=jnp.complex128)


def energy(hamiltonian, state):
    """ Compute the energy level of a state under given hamiltonian.

    E = <s| H |s>

    Args:
        hamiltonian: jnp.array, of shape (2 ** qubit, 2 ** qubit),
            hamiltonian matrix
        state: jnp.array, of shape (2 ** qubit,) a state vector
    Returns:
        jnp.scalar, energy
    """
    return jnp.real(state.T.conj() @ hamiltonian @ state)


def fidelity(state, target_state):
    """ Compute the fidelity between two states. """
    return jnp.abs(state.T.conj() @ target_state) ** 2


def bandwidth(ham_matrix):
    eigval, _ = onp.linalg.eigh(ham_matrix)
    bandwidth = jnp.real(jnp.max(eigval) - jnp.min(eigval))
    return bandwidth


def ising_hamiltonian(n_qubits, g, h):
    """ Construct the hamiltonian matrix of Ising model.

    Args:
        n_qubits: int, Number of qubits
        g: float, Transverse magnetic field
        h: float, Longitudinal magnetic field
    """
    ham_matrix = 0

    # Nearest-neighbor interaction
    spin_coupling = jnp.kron(PauliBasis[3], PauliBasis[3])

    for i in range(n_qubits - 1):
        ham_matrix -= jnp.kron(jnp.kron(jnp.eye(2 ** i), spin_coupling),
                               jnp.eye(2 ** (n_qubits - 2 - i)))
    ham_matrix -= jnp.kron(jnp.kron(PauliBasis[3], jnp.eye(2 ** (n_qubits - 2))),
                           PauliBasis[3])  # Periodic B.C

    # Transverse magnetic field
    for i in range(n_qubits):
        ham_matrix -= g * jnp.kron(jnp.kron(jnp.eye(2 ** i), PauliBasis[1]),
                                   jnp.eye(2 ** (n_qubits - 1 - i)))

    # Longitudinal magnetic field
    for i in range(n_qubits):
        ham_matrix -= h * jnp.kron(jnp.kron(jnp.eye(2 ** i), PauliBasis[3]),
                                   jnp.eye(2 ** (n_qubits - 1 - i)))
    return ham_matrix


def SYK_hamiltonian(rng, n_qubits):
    """ Construct the hamiltonian matrix of Ising model.

    Args:
        n_qubits: int, Number of qubits
        g: float, Transverse magnetic field
        h: float, Longitudinal magnetic field
    """
    # Construct the gamma matrices for SO(2 * n_qubits) Clifford algebra
    gamma_matrices, n_gamma = [], 2 * n_qubits

    for k in range(n_gamma):
        temp = jnp.eye(1)

        for j in range(k // 2):
            temp = jnp.kron(temp, PauliBasis[3])

        if k % 2 == 0:
            temp = jnp.kron(temp, PauliBasis[1])
        else:
            temp = jnp.kron(temp, PauliBasis[2])

        for i in range(int(n_gamma / 2) - (k // 2) - 1):
            temp = jnp.kron(temp, PauliBasis[0])

        gamma_matrices.append(temp)

    # Number of SYK4 interaction terms
    n_terms = int(factorial(n_gamma) / factorial(4) / factorial(n_gamma - 4))

    # SYK4 random coupling
    couplings = jax.random.normal(key=rng, shape=(n_terms,), dtype=jnp.float64) * jnp.sqrt(6 / (n_gamma ** 3))

    ham_matrix = 0
    for idx, (x, y, w, z) in enumerate(combinations(range(n_gamma), 4)):
        ham_matrix += (couplings[idx] / 4) * jnp.linalg.multi_dot(
            [gamma_matrices[x], gamma_matrices[y], gamma_matrices[w], gamma_matrices[z]])

    return ham_matrix


def memory_efficient_hessian(f):
    def hessian_fn(x):
        _, hvp = jax.linearize(jax.grad(f), x)
        hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = jnp.eye(jnp.prod(x.shape)).reshape(-1, *x.shape)
        return jnp.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)

    return hessian_fn


def add_circuit_arguments(parser):
    group = parser.add_argument_group('Circuit Options')
    group.add_argument('--n-qubits', type=int, metavar='N', required=True,
                       help='Number of qubits')
    group.add_argument('--n-layers', type=int, metavar='N', required=True,
                       help='Number of alternating layers')
    group.add_argument('--rot-axis', type=str, metavar='R', required=True,
                       choices=['x', 'y', 'z'],
                       help='Direction of rotation gates.')


def add_optimizer_arguments(parser):
    group = parser.add_argument_group('Optimization Options')
    group.add_argument(
        '--train-steps', type=int, metavar='N', default=1000,
        help='Number of training steps. (Default: 1000)')
    group.add_argument(
        '--lr', type=float, metavar='LR', default=0.05,
        help='Initial value of learning rate. (Default: 0.05)')
    group.add_argument(
        '--optimizer-name', type=str, metavar='NAME', default='adam',
        help='Optimizer name. Supports: adam, nesterov, sgd (Default: adam)')
    group.add_argument(
        '--optimizer-args', type=str, metavar='STR', default=None,
        help='Additional arguments for the chosen optimizer.\n'
             'For instance, --optimizer-name=nesterov --optimizer-args="mass:0.1"'
             ' or --optimizer-name=adam --optimizer-args="eps:1e-8,b1:0.9,b2:0.999"'
             ' (Default: None)')
    group.add_argument(
        '--scheduler-name', type=str, metavar='NAME', default='constant',
        help=f'Scheduler name. Supports: {supported_schedulers()} '
             f'(Default: constant)')
    group.add_argument(
        '--checkpoint-path', type=str, metavar='PATH', default=None,
        help='A checkpoint file path to resume')
    group.add_argument(
        '--log-every', type=int, metavar='N', default=1,
        help='Logging every N steps. (Default: 1)')
    group.add_argument(
        '--no-jacfwd', dest='use_jacfwd', action='store_false',
        help='Disable the forward mode gradient computation (jacfwd)'
             'and use the reversed mode jax.grad (equivalently jax.jacrev).')
