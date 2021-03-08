""" Basic gate operations for quantum circuit.

This module is copied and modified from qutip.qip.gates.py.

"""

import jax.numpy as jnp


def gate_expand_1toN(U, N, target):
    """
    Create a JAX array representing a 1-qubit gate acting on a system with N qubits.
    Parameters
    ----------
    U : 2 X 2 unitary matrix
        The one-qubit gate
    N : integer
        The number of qubits in the target space.
    target : integer
        The index of the target qubit.
    Returns
    -------
    gate : (2 ** N) X (2 ** N) unitary matrix
        Quantum object representation of N-qubit gate.
    """

    if N < 1:
        raise ValueError("integer N must be larger or equal to 1")

    if target >= N:
        raise ValueError("target must be integer < integer N")

    return jnp.kron(jnp.kron(jnp.eye(2 ** target), U),
                    jnp.eye(2 ** (N - target - 1)))


def gate_expand_2toN(U, N, control=None, target=None):
    """
    Create a Qobj representing a two-qubit gate that act on a system with N
    qubits.
    Parameters
    ----------
    U : 4 X 4 unitary matrix
        The two-qubit gate
    N : integer
        The number of qubits in the target space.
    control : integer
        The index of the control qubit.
    target : integer
        The index of the target qubit.
    Returns
    -------
    gate :  (2 ** N) X (2 ** N) unitary matrix
        Quantum object representation of N-qubit gate.
    """

    if control is None or target is None:
        raise ValueError("Specify value of control and target")

    if N < 2:
        raise ValueError("integer N must be larger or equal to 2")

    if control >= N or target >= N:
        raise ValueError("control and not target must be integer < integer N")

    if control == target:
        raise ValueError("target and not control cannot be equal")

    p = list(range(N))

    if target == 0 and control == 1:
        p[control], p[target] = p[target], p[control]

    elif target == 0:
        p[1], p[target] = p[target], p[1]
        p[1], p[control] = p[control], p[1]

    else:
        p[1], p[target] = p[target], p[1]
        p[0], p[control] = p[control], p[0]

    matrix = jnp.kron(U, jnp.eye(2 ** (N - 2)))
    matrix = matrix.reshape(*([2 for _ in range(2 * N)]))
    matrix = matrix.transpose(p + [i + N for i in p])
    matrix = matrix.reshape(2 ** N, 2 ** N)

    return matrix


def x_gate(N=None, target=0):
    """Pauli-X gate or sigmax operator.
    Returns
    -------
    result : (2 ** N) X (2 ** N) unitary matrix
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the x-axis.
    """
    if N is not None:
        return gate_expand_1toN(x_gate(), N, target)
    return jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)


def y_gate(N=None, target=0):
    """Pauli-Y gate or sigmay operator.
    Returns
    -------
    result : (2 ** N) X (2 ** N) unitary matrix
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the y-axis.
    """
    if N is not None:
        return gate_expand_1toN(y_gate(), N, target)
    return jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)


def cy_gate(N=None, control=0, target=1):
    """Controlled Y gate.
    Returns
    -------
    result : (2 ** N) X (2 ** N) unitary matrix
        Quantum object for operator describing the rotation.
    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cy_gate(), N, control, target)
    return jnp.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, -1j],
                      [0, 0, 1j, 0]], dtype=jnp.complex64)


def z_gate(N=None, target=0):
    """Pauli-Z gate or sigmaz operator.
    Returns
    -------
    result : (2 ** N) X (2 ** N) unitary matrix
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the z-axis.
    """
    if N is not None:
        return gate_expand_1toN(z_gate(), N, target)
    return jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)


def cz_gate(N=None, control=0, target=1):
    """Controlled Z gate.
    Returns
    -------
    result : (2 ** N) X (2 ** N) unitary matrix
        Quantum object for operator describing the rotation.
    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cz_gate(), N, control, target)
    return jnp.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1j]], dtype=jnp.complex64)


def rx(phi, N=None, target=0):
    """Single-qubit rotation for operator sigmax with angle phi.
    Returns
    -------
    result : (2 ** N) X (2 ** N) unitary matrix
        Quantum object for operator describing the rotation.
    """
    if N is not None:
        return gate_expand_1toN(rx(phi), N, target)
    return jnp.array([[jnp.cos(phi / 2), -1j * jnp.sin(phi / 2)],
                      [-1j * jnp.sin(phi / 2), jnp.cos(phi / 2)]], dtype=jnp.complex64)


def ry(phi, N=None, target=0):
    """Single-qubit rotation for operator sigmay with angle phi.
    Returns
    -------
    result : (2 ** N) X (2 ** N) unitary matrix
        Quantum object for operator describing the rotation.
    """
    if N is not None:
        return gate_expand_1toN(ry(phi), N, target)
    return jnp.array([[jnp.cos(phi / 2), -jnp.sin(phi / 2)],
                      [jnp.sin(phi / 2), jnp.cos(phi / 2)]], dtype=jnp.complex64)


def rz(phi, N=None, target=0):
    """Single-qubit rotation for operator sigmaz with angle phi.
    Returns
    -------
    result : (2 ** N) X (2 ** N) unitary matrix
        Quantum object for operator describing the rotation.
    """
    if N is not None:
        return gate_expand_1toN(rz(phi), N, target)
    return jnp.array([[jnp.exp(-1j * phi / 2), 0],
                      [0, jnp.exp(1j * phi / 2)]], dtype=jnp.complex64)


def sqrtnot(N=None, target=0):
    """Single-qubit square root NOT gate.
    Returns
    -------
    result : (2 ** N) X (2 ** N) unitary matrix
        Quantum object for operator describing the square root NOT gate.
    """
    if N is not None:
        return gate_expand_1toN(sqrtnot(), N, target)
    return jnp.array([[0.5 + 0.5j, 0.5 - 0.5j],
                      [0.5 - 0.5j, 0.5 + 0.5j]], dtype=jnp.complex64)


def cnot(N=None, control=0, target=1):
    """Controlled Z gate.
    Returns
    -------
    result : (2 ** N) X (2 ** N) unitary matrix
        Quantum object for operator describing the rotation.
    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cz_gate(), N, control, target)
    return jnp.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], dtype=jnp.complex64)


def csign(N=None, control=0, target=1):
    """
    Quantum object representing the CSIGN gate.
    Returns
    -------
    csign_gate : (2 ** N) X (2 ** N) unitary matrix
        Quantum object representation of CSIGN gate
    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(csign(), N, control, target)
    return jnp.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]], dtype=jnp.complex64)
