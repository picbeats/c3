"""
Synthetic C3 model learning

Construct two models, a 'wrong' one and a 'correct' one. First construct
open loop pulses with the wrong model then calibrate them against the real
model, simulating an experiment. Finally try to match the wrong model to the
calibration data and recover the real one.
"""

import numpy as np
import tensorflow as tf
import c3po.hamiltonians as hamiltonians
from c3po.utils import log_setup
from c3po.component import Quantity as Qty
from c3po.simulator import Simulator as Sim
from c3po.experiment import Experiment as Exp
from c3po.tf_utils import tf_abs
from c3po.qt_utils import basis, xy_basis, perfect_gate
from single_qubit import create_chip_model, create_generator, create_gates

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

logdir = log_setup("/tmp/c3logs/")

# System
qubit_freq = Qty(
    value=5.1173e9 * 2 * np.pi,
    min=5.1e9 * 2 * np.pi,
    max=5.14e9 * 2 * np.pi,
    unit='Hz 2pi'
)
qubit_anhar = Qty(
    value=-315.513734e6 * 2 * np.pi,
    min=-330e6 * 2 * np.pi,
    max=-300e6 * 2 * np.pi,
    unit='Hz 2pi'
)
qubit_lvls = 4
drive_ham = hamiltonians.x_drive
v_hz_conversion = Qty(
    value=1,
    min=0.9,
    max=1.1,
    unit='rad/V'
)

qubit_freq_wrong = Qty(
    value=5.12e9 * 2 * np.pi,
    min=5.1e9 * 2 * np.pi,
    max=5.14e9 * 2 * np.pi,
    unit='Hz 2pi'
)

qubit_lvls = 4
drive_ham = hamiltonians.x_drive

t_final = Qty(
    value=10e-9,
    min=5e-9,
    max=15e-9,
    unit='s'
)
rise_time = Qty(
    value=0.1e-9,
    min=0.0e-9,
    max=0.2e-9,
    unit='s'
)

# Define the ground state
qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1
ket_0 = tf.constant(qubit_g, tf.complex128)
bra_0 = tf.constant(qubit_g.T, tf.complex128)

# Simulation variables
sim_res = 60e9
awg_res = 1.2e9

# Create system
model_wrong = create_chip_model(
    qubit_freq_wrong, qubit_anhar, qubit_lvls, drive_ham
)
model_right = create_chip_model(
    qubit_freq, qubit_anhar, qubit_lvls, drive_ham
)
gen_wrong = create_generator(
    sim_res, awg_res, v_hz_conversion, logdir=logdir,
    rise_time=rise_time
)
gen_right = create_generator(
    sim_res, awg_res, v_hz_conversion, logdir=logdir, rise_time=rise_time
)
gates = create_gates(
    t_final=t_final,
    v_hz_conversion=v_hz_conversion,
    qubit_freq=qubit_freq_wrong,
    qubit_anhar=qubit_anhar
)

# gen.devices['awg'].options = 'drag'

# Simulation class and fidelity function
exp_wrong = Exp(
    model_wrong,
    gen_wrong
)
sim_wrong = Sim(exp_wrong, gates)
a_q = model_wrong.ann_opers[0]

exp_right = Exp(model_right, gen_right)
sim_right = Sim(exp_right, gates)
a_q = model_right.ann_opers[0]

# Define states
# Define states & unitaries
ket_0 = tf.constant(basis(qubit_lvls, 0), dtype=tf.complex128)
bra_2 = tf.constant(basis(qubit_lvls, 2).T, dtype=tf.complex128)
bra_yp = tf.constant(xy_basis(qubit_lvls, 'yp').T, dtype=tf.complex128)
X90p = tf.constant(perfect_gate(qubit_lvls, 'X90p'), dtype=tf.complex128)


# TODO move fidelity experiments elsewhere
def state_transfer_infid(U_dict: dict):
    U = U_dict['X90p']
    ket_actual = tf.matmul(U, ket_0)
    overlap = tf_abs(tf.matmul(bra_yp, ket_actual))
    infid = 1 - overlap
    return infid


def unitary_infid(U_dict: dict):
    U = U_dict['X90p']
    unit_fid = tf_abs(
        tf.linalg.trace(tf.matmul(U, tf.linalg.adjoint(X90p))) / 2
    )**2
    infid = 1 - unit_fid
    return infid


def pop_leak(U_dict: dict):
    U = U_dict['X90p']
    ket_actual = tf.matmul(U, ket_0)
    overlap = tf_abs(tf.matmul(bra_2, ket_actual))
    return overlap


def match_calib(
    exp_params: list,
    exp_opt_map: list,
    gateset_values: list,
    gateset_opt_map: list,
    seq: list,
    fid: np.float64
):
    exp_wrong.set_parameters(exp_params, exp_opt_map)
    sim_wrong.gateset.set_parameters(gateset_values, gateset_opt_map)
    U_dict = sim_wrong.get_gates()
    fid_sim = unitary_infid(U_dict)
    diff = fid_sim - fid
    return diff


def infid_sim(params):
    exp_wrong.set_parameters(params, exp_opt_map)
    U_dict = sim_wrong.get_gates()
    return unitary_infid(U_dict)


# Optimizer object
opt_map = [
    [('X90p', 'd1', 'gauss', 'amp')],
    [('X90p', 'd1', 'gauss', 'freq_offset')],
    [('X90p', 'd1', 'gauss', 'xy_angle')],
    # [('X90p', 'd1', 'gauss', 'delta')]
]

exp_opt_map = [
    ('Q1', 'freq'),
    # ('Q1', 'anhar'),
    # ('Q1', 't1'),
    # ('Q1', 't2star'),
    ('v_to_hz', 'V_to_Hz'),
    # ('resp', 'rise_time')
]


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data
X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
infid = np.zeros((X.shape[0], Y.shape[0]))
for ii in range(X.shape[0]):
    for jj in range(Y.shape[0]):
        params = [X[ii], Y[jj]]
        infid[ii][jj] = infid_sim(params)
X, Y = np.meshgrid(X, Y)
# Plot the surface.
surf = ax.plot_surface(X, Y, infid, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
