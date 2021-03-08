import argparse

import jax

import expmgr
import qnnops

parser = argparse.ArgumentParser('Expressibility Test')
parser.add_argument('--n-qubits', type=int, metavar='N', required=True,
                    help='Number of qubits')
parser.add_argument('--n-layers', type=int, metavar='N', required=True,
                    help='Number of alternating layers')
parser.add_argument('--rot-axis', type=str, metavar='R', default='y',
                    choices=['x', 'y', 'z'],
                    help='Direction of rotation gates.')
parser.add_argument('--train-steps', type=int, metavar='N', default=int(1e3),
                    help='Number of training steps.')
parser.add_argument('--lr', type=float, metavar='LR', default=0.05,
                    help='Initial value of learning rate.')
parser.add_argument('--seed', type=int, metavar='N', required=True,
                    help='Random seed. For reproducibility, the value is set explicitly.')
parser.add_argument('--exp-name', type=str, metavar='NAME', default=None,
                    help='Experiment name. If None, the following format will be used as '
                         'the experiment name: Q{n_qubits}L{n_layers}R{rot_axis}BS{block_size} - S{seed} - LR{args.lr}')
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
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='Resume from the last checkpoint under `results/{exp_name}/`.'
                         'This option works only with --no-time-tag.')
parser.add_argument('--no-jit', dest='use_jit', action='store_false',
                    help='Disable jit option to loss function.')
parser.add_argument('--no-time-tag', dest='time_tag', action='store_false',
                    help='Omit the time tag from experiment name.')
parser.add_argument('--use-jacfwd', dest='use_jacfwd', action='store_true',
                    help='Enable the forward mode gradient computation (jacfwd).')
parser.add_argument('--version', type=int, default=1, choices=[1, 2],
                    help='qnnops version (Default: 1)')

args = parser.parse_args()
seed = args.seed
n_qubits, n_layers, rot_axis = args.n_qubits, args.n_layers, args.rot_axis
block_size = args.n_qubits
exp_name = args.exp_name or f'Q{n_qubits}L{n_layers}R{rot_axis}BS{block_size} - S{seed} - LR{args.lr}'
expmgr.init(project='expressibility', name=exp_name, config=args)

target_state = qnnops.create_target_states(n_qubits, 1, seed=seed)
expmgr.log_array(target_state=target_state)
expmgr.save_array('target_state.npy', target_state)


def circuit(params):
    return qnnops.alternating_layer_ansatz(
        params, n_qubits=n_qubits, block_size=block_size, n_layers=n_layers, rot_axis=rot_axis)


def loss_fn(params):
    ansatz_state = circuit(params)
    return qnnops.state_norm(ansatz_state - target_state) / (2 ** n_qubits)


rng = jax.random.PRNGKey(seed)
_, init_params = qnnops.initialize_circuit_params(rng, n_qubits, n_layers)
trained_params, _ = qnnops.train_loop(
    loss_fn, init_params, args.train_steps, args.lr,
    optimizer_name=args.optimizer_name, optimizer_args=args.optimizer_args,
    scheduler_name=args.scheduler_name,
    checkpoint_path=args.checkpoint_path,
    use_jit=args.use_jit,
    use_jacfwd=args.use_jacfwd
)

optimized_state = circuit(trained_params)
expmgr.log_array(optimized_state=optimized_state)
expmgr.save_array('optimized_state.npy', optimized_state)

expmgr.save_config(args)
