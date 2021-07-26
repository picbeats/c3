# System imports
import copy
import numpy as np
import time
import itertools
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns
import multiprocessing

# Main C3 objects
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as PMap
from c3.experiment import Experiment as Exp
from c3.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
from c3.generator.devices import Device
import c3.signal.gates as gates
from c3.signal.gates import Instruction
import c3.libraries.chip as chip
import c3.signal.pulse as pulse
import c3.libraries.tasks as tasks

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.fidelities as fidelities
import c3.libraries.envelopes as envelopes
import c3.utils.qt_utils as qt_utils
import c3.utils.tf_utils as tf_utils
import c3.utils.noise_utils as ns_utils

###ssh forwarding
matplotlib.use("tkagg")  # Or any other X11 back-end
###user defs
class ChargeBias(Device):
    def __init__(self, omega, amplitude, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 0)
        self.outputs = props.pop("outputs", 1)
        self.omega = omega
        self.amplitude = amplitude

    def process(self, instr: Instruction, chan: str):
        # charge_noise=ns_utils.load_spectrum("data/data_charge.dat")
        # charge_noise[:,0]=2*np.pi*charge_noise[:,0]
        # N=500#int(max(charge_noise[:,0])/min(charge_noise[:,0]))
        # freqs, times, ngs = ns_utils.generate_time_shot(charge_noise,N)
        # print("times:" , times)
        # 27337017192.233242#
        times = np.linspace(0, 400e-9, 200)
        ngs = self.amplitude * np.cos(self.omega * times)
        signal = {
            "NG": ngs,
            "ts": times,
        }
        return signal


###propagtion because the one c3 does not work
def tf_propagation_vectorized(h0, hks, cflds_t, dt):
    dt = tf.cast(dt, dtype=tf.complex128)
    if hks is not None and cflds_t is not None:
        cflds_t = tf.cast(cflds_t, dtype=tf.complex128)
        hks = tf.cast(hks, dtype=tf.complex128)
        cflds = tf.expand_dims(tf.expand_dims(cflds_t, 2), 3)
        hks = tf.expand_dims(hks, 1)
        if len(h0.shape) < 3:
            h0 = tf.expand_dims(h0, 0)
        prod = cflds * hks
        h = h0 + tf.reduce_sum(prod, axis=0)
    else:
        h = tf.cast(h0, tf.complex128)
    dUs = []
    delta = dt[1]
    for i in range(len(dt)):
        dh = -1.0j * h[i] * delta
        dU = tf.linalg.expm(dh)
        dUs.append(dU)
    return dUs

    def plot_dynamics(exp, psi_init, seq, goal=-1, dUs=None, dt=None, w=None):
        """
        Plotting code for time-resolved populations.

        Parameters
        ----------
        psi_init: tf.Tensor
            Initial state or density matrix.
        seq: list
            List of operations to apply to the initial state.
        goal: tf.float64
            Value of the goal function, if used.
        debug: boolean
            If true, return a matplotlib figure instead of saving.
        """
        model = exp.pmap.model
        model.use_FR = False
        model.lindbladian = False
        ###does not work dUs = exp.compute_propagators() #partial_propagators
        psi_t = psi_init.numpy()
        pop_t = exp.populations(psi_t, model.lindbladian)
        # tf.reshape(pop_t, (21,))
        for du in dUs:
            psi_t = np.matmul(du.numpy(), psi_t)
            pops = exp.populations(psi_t, model.lindbladian)
            pop_t = np.append(pop_t, pops, axis=1)
        fig, axs = plt.subplots(1, 1)
        ts = dt
        dt = ts[1] - ts[0]
        ts = np.linspace(0.0, dt * pop_t.shape[1], pop_t.shape[1])
        # axs.plot(ts / 1e-9, pop_t.T)
        axs.plot(ts / 1e-9, pop_t.T[:, 0])
        axs.plot(ts / 1e-9, pop_t.T[:, 1])
        # return pop_t.T[:,0]

        axs.plot(ts / 1e-9, pop_t.T[:, 2])
        axs.grid(linestyle="--")
        axs.tick_params(direction="in", left=True, right=True, top=True, bottom=True)
        axs.set_xlabel("Time [ns]")
        axs.set_ylabel("Population")
        plt.legend(model.state_labels)
        plt.show()
        # plt.savefig('test.pdf')


def collect_resonance(exp, psi_init, seq, goal=-1, dUs=None, dt=None):
    model = exp.pmap.model
    model.use_FR = False
    model.lindbladian = False
    psi_t = psi_init.numpy()
    pop_t = exp.populations(psi_t, model.lindbladian)
    for du in dUs:
        psi_t = np.matmul(du.numpy(), psi_t)
        pops = exp.populations(psi_t, model.lindbladian)
        pop_t = np.append(pop_t, pops, axis=1)
    ts = dt
    dt = ts[1] - ts[0]
    ts = np.linspace(0.0, dt * pop_t.shape[1], pop_t.shape[1])
    return pop_t.T[:, 1]  # ,pop_t.T[-1,1]


############### Creating qubit ###################
qubit_lvls = 21
q1_ec = 270.8e6  # 5ghz 300Mhz  sqrt(EC*EJ) 50J
q1_ej = 22.25e9
q1_temp = 60e-3
q1_ng = 0
q1_asym = 0
q1_redflux = 0

q1 = chip.CooperPairBox(
    name="Q1",
    desc="Qubit 1",
    EC=Qty(value=q1_ec, min_val=200e6, max_val=400e6, unit="Hz 2pi"),
    EJ=Qty(value=q1_ej, min_val=1e9, max_val=30e9, unit="Hz 2pi"),
    hilbert_dim=qubit_lvls,
    NG=Qty(value=q1_ng, min_val=-1, max_val=1, unit=""),
    Asym=Qty(value=q1_asym, min_val=-1, max_val=1, unit=""),
    Reduced_Flux=Qty(value=q1_redflux, min_val=-1, max_val=1, unit=""),
)

############ Generate Model ###################
model = Mdl([q1], [], [])
model.use_FR = False
model.lindbladian = False
############# For plot dynamics ################
# signal=generator.generate_signals(Instruction(channels=["Q1"]))
# HCB=q1.get_Hamiltonian(signal=signal["Q1"])
# Hs=[]
# H_0=q1.get_Hamiltonian()
# e,v=tf.linalg.eigh(H_0)
# print(e[1]-e[0])

# for h in HCB:
#    tmp=tf.linalg.inv(v)@h@v
#    Hs.append(tmp)
# dt=signal["Q1"]["ts"]
# dUs=tf_propagation_vectorized(Hs, hks=None, cflds_t=None, dt=dt)
############# generate instructions (here do nothing) ####
t_final = 3e-9
nodrive_env = pulse.Envelope(
    name="no_drive",
    params={
        "t_final": Qty(
            value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
        )
    },
    shape=envelopes.no_drive,
)

########### adding instructions #################
identity_q1 = gates.Instruction(
    name="identity", targets=[0], t_start=0.0, t_end=t_final, channels=["Q1"]
)
identity_q1.add_component(nodrive_env, "Q1")
single_q_gates = [identity_q1]


##Dynamics
psi_init = [[0] * 21]  ###thermal states als initials
psi_init[0][0] = 1
psi_init[0][1] = 0
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
# init_state = v @ init_state
barely_a_seq = ["identity"]

# plot_dynamics(exp, init_state, barely_a_seq, dUs=dUs, dt=dt)
omega = np.linspace(
    41847055724.892105 - 2 * np.pi * 50e6, 41847055724.892105 + 2 * np.pi * 50e6, 100
)
amplitude = np.linspace(0, 0.1, 100)
H_0 = q1.get_Hamiltonian()
e, v = tf.linalg.eigh(H_0)


def calc_population(k):
    i = k % 10
    j = int(k / 10)
    w = omega[j]
    a = amplitude[i]
    generator = Gnr(
        devices={"ChargeBias": ChargeBias(w, a, name="ChargeBias")},
        chains={"Q1": ["ChargeBias"]},
    )
    parameter_map = PMap(instructions=single_q_gates, model=model, generator=generator)
    # exp = Exp(pmap=parameter_map)
    # exp.use_control_fields=False
    signal = generator.generate_signals(Instruction(channels=["Q1"]))
    HCB = q1.get_Hamiltonian(signal=signal["Q1"])
    Hs = []
    for h in HCB:
        tmp = tf.linalg.inv(v) @ h @ v
        Hs.append(tmp)
    dt = signal["Q1"]["ts"]
    dUs = tf_propagation_vectorized(Hs, hks=None, cflds_t=None, dt=dt)
    return collect_resonance(exp, init_state, barely_a_seq, dUs=dUs, dt=dt)


pool = multiprocessing.Pool(8)
data = zip(*pool.map(calc_population, range(100)))
data = np.array(data)
np.savetxt(
    "chevron_parallel=" + str(q1_ec) + "_ej=" + str(q1_ej) + "_pop1.csv",
    data,
    delimiter=",",
)

############
##SPAM
# m00_q1 = 0.8  # Prop to read qubit 1 state 0 as 0
# m01_q1 = 0.2  # Prop to read qubit 1 state 0 as 1
# one_zeros = np.array([0] * qubit_lvls)
# zero_ones = np.array([1] * qubit_lvls)
# one_zeros[0] = 1
# zero_ones[0] = 0
# val1 = one_zeros * m00_q1 + zero_ones * m01_q1
# min_val = one_zeros * 0.8 + zero_ones * 0.0
# max_val = one_zeros * 1.0 + zero_ones * 0.2
# confusion_row1 = Qty(value=val1, min_val=min_val, max_val=max_val, unit="")
# conf_matrix = tasks.ConfusionMatrix(Q1=confusion_row1)
#
##comments from Christian
##basteln drive und readout vllt in charge basis verlegen
##drive Vg n*(a+a^\dagger e ) #evtl drive durch cavity wenn transmon und cavity frequenz aufpassen
##temperatur nicht im thermischen Gleichgewicht 1803.00476.pdf,1409.6031
##effective temperatur: Linblad operatoren
##SPAM readout 80% readout fidelity -> später
##T~~60mk (0-150))
##set rotating frame yes:
#
# init_temp=60e-3
# init_ground = tasks.InitialiseGround(
#    init_temp=Qty(
#        value=init_temp,
#        min_val=-0.001,
#        max_val=0.22,
#        unit='K'
#    )
# )
#
##generate Model
# model = Mdl(
#    [q1], # Individual, self-contained components
#    [drive],  # Interactions between components
#    [conf_matrix, init_ground] # SPAM processing
# )
#
# model.set_dressed(True)
# model.set_lindbladian(False)
#
##Control signals
#
# sim_res=750e6 #lo rode_schwarz Hz
# awg_res=2.4e9#
#
# lo = devices.LO(name='lo',resolution=sim_res)
# awg = devices.AWG(name='awg',resolution=awg_res)
# mixer = devices.Mixer(name='mixer')
#
# resp = devices.Response(
#    name='resp',
#    rise_time=Qty(
#        value=0.3e-9,
#        min_val=0.05e-9,
#        max_val=0.6e-9,
#        unit='s'
#    ),
#    resolution=sim_res
# )
#
# dig_to_an = devices.DigitalToAnalog(
#    name="dac",
#    resolution=sim_res
# )
##Volts to hertz converter
# v2hz = 1e9
# v_to_hz = devices.VoltsToHertz(
#    name='v_to_hz',
#    V_to_Hz=Qty(
#        value=v2hz,
#        min_val=0.9e9,
#        max_val=1.1e9,
#        unit='Hz/V'
#    )
# )
#
##Parity_noise=devices.Pink_Noise(
##        name="parity_noise",
##        noise_strenght=0.001, #look how much is needed
##        bbfl_num=1
##        )
#
# Charge_noise=devices.Colored_Noise(
#        name="charge_noise",
#        noise_strength=Qty(value=0.01,min_val=0,max_val=0.02,unit=""),
#        bfl_num=Qty(value=5,min_val=0,max_val=10,unit="")
#        )
#
#
#
##Generator combining signal generation and assigning signal chain to control line
# generator = Gnr(
#        devices={
#            "LO": devices.LO(name='lo', resolution=sim_res, outputs=1),
#            "AWG": devices.AWG(name='awg', resolution=awg_res, outputs=1),
#            "DigitalToAnalog": devices.DigitalToAnalog(
#                name="dac",
#                resolution=sim_res,
#                inputs=1,
#                outputs=1
#            ),
#            "Response": devices.Response(
#                name='resp',
#                rise_time=Qty(
#                    value=0.3e-9,
#                    min_val=0.05e-9,
#                    max_val=0.6e-9,
#                    unit='s'
#                ),
#                resolution=sim_res,
#                inputs=1,
#                outputs=1
#            ),
#            "Mixer": devices.Mixer(name='mixer', inputs=2, outputs=1),
#            "VoltsToHertz": devices.VoltsToHertz(
#                name='v_to_hz',
#                V_to_Hz=Qty(
#                    value=1e9,
#                    min_val=0.9e9,
#                    max_val=1.1e9,
#                    unit='Hz/V'
#                ),
#                inputs=1,
#                outputs=1
#            ),
#            "Charge_noise": devices.Colored_Noise(
#                name="charge_noise",
#                noise_strength=Qty(value=0.01,min_val=0,max_val=0.02,unit=""),
#                bfl_num=Qty(value=5,min_val=0,max_val=10,unit=""),
#                inputs=0,
#                outputs=1
#                )
#        },
#        chains= {
#            "d1": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"],
#            "d2": ["charge_noise"]
#        }
#    )
#
##gate  set
# t_final = 7e-8   # Time for single qubit gates
# sideband = 50e6
# gauss_params_single = {
#    'amp': Qty(
#        value=0.5,
#        min_val=0.4,
#        max_val=0.6,
#        unit="V"
#    ),
#    't_final': Qty(
#        value=t_final,
#        min_val=0.5 * t_final,
#        max_val=1.5 * t_final,
#        unit="s"
#    ),
#    'sigma': Qty(
#        value=t_final / 4,
#        min_val=t_final / 8,
#        max_val=t_final / 2,
#        unit="s"
#    ),
#    'xy_angle': Qty(
#        value=0.0,
#        min_val=-0.5 * np.pi,
#        max_val=2.5 * np.pi,
#        unit='rad'
#    ),
#    'freq_offset': Qty(
#        value=-sideband - 3e6 ,
#        min_val=-56 * 1e6 ,
#        max_val=-52 * 1e6 ,
#        unit='Hz 2pi'
#    ),
#    'delta': Qty(
#        value=-1,
#        min_val=-5,
#        max_val=3,
#        unit=""
#    )
# }
##gaussian puls
# gauss_env_single = pulse.Envelope(
#    name="gauss",
#    desc="Gaussian comp for single-qubit gates",
#    params=gauss_params_single,
#    shape=envelopes.gaussian_nonorm
# )
#
##nodrive_env
# nodrive_env = pulse.Envelope(
#    name="no_drive",
#    params={
#        't_final': Qty(
#            value=t_final,
#            min_val=0.5 * t_final,
#            max_val=1.5 * t_final,
#            unit="s"
#        )
#    },
#    shape=envelopes.no_drive
# )
#
#
###specify drive tones
# lo_freq_q1 = 5e9  + sideband
# carrier_parameters = {
#    'freq': Qty(
#        value=lo_freq_q1,
#        min_val=4.5e9 ,
#        max_val=6e9 ,
#        unit='Hz 2pi'
#    ),
#    'framechange': Qty(
#        value=0.0,
#        min_val= -np.pi,
#        max_val= 3 * np.pi,
#        unit='rad'
#    )
# }
# carr = pulse.Carrier(
#    name="carrier",
#    desc="Frequency of the local oscillator",
#    params=carrier_parameters
# )
#
##add instructions
# rx90p_q1 = gates.Instruction(
#    name="rx90p", #targets=[0],
#    t_start=0.0, t_end=t_final, channels=["d1","d2"]
# )
# rx90p_q1.add_component(gauss_env_single, "d1")
# rx90p_q1.add_component(carr, "d1")
#
##lo_freq_q2 = 5.6e9  + sideband
##carr_2 = copy.deepcopy(carr)
##carr_2.params['freq'].set_value(lo_freq_q2)
##
##rx90p_q1.add_component(nodrive_env, "d2")
##rx90p_q1.add_component(copy.deepcopy(carr_2), "d2")
##rx90p_q1.comps["d2"]["carrier"].params["framechange"].set_value(
##    (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
##)
##
# ry90p_q1 = copy.deepcopy(rx90p_q1)
# ry90p_q1.name = "ry90p"
# rx90m_q1 = copy.deepcopy(rx90p_q1)
# rx90m_q1.name = "rx90m"
# ry90m_q1 = copy.deepcopy(rx90p_q1)
# ry90m_q1.name = "ry90m"
# ry90p_q1.comps['d1']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
# rx90m_q1.comps['d1']['gauss'].params['xy_angle'].set_value(np.pi)
# ry90m_q1.comps['d1']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)
# single_q_gates = [rx90p_q1, ry90p_q1, rx90m_q1, ry90m_q1]
#
##collecting all Parameters togheter
# parameter_map = PMap(instructions=single_q_gates, model=model, generator=generator)
#
##creating experiment
# exp = Exp(pmap=parameter_map)
# unitaries = exp.get_gates()
#
##Dynamics
# psi_init = [[0] * 11]
# psi_init[0][9] = 1
# init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
#
# barely_a_seq=['rx90p']
# def plot_dynamics(exp, psi_init, seq, goal=-1):
#        """
#        Plotting code for time-resolved populations.
#
#        Parameters
#        ----------
#        psi_init: tf.Tensor
#            Initial state or density matrix.
#        seq: list
#            List of operations to apply to the initial state.
#        goal: tf.float64
#            Value of the goal function, if used.
#        debug: boolean
#            If true, return a matplotlib figure instead of saving.
#        """
#        model = exp.pmap.model
#        dUs = exp.dUs #partial_propagators
#        psi_t = psi_init.numpy()
#        pop_t = exp.populations(psi_t, model.lindbladian)
#        for gate in seq:
#            for du in dUs[gate]:
#                psi_t = np.matmul(du.numpy(), psi_t)
#                pops = exp.populations(psi_t, model.lindbladian)
#                pop_t = np.append(pop_t, pops, axis=1)
#
#        fig, axs = plt.subplots(1, 1)
#        ts = exp.ts
#        dt = ts[1] - ts[0]
#        ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
#        axs.plot(ts / 1e-9, pop_t.T)
#        axs.grid(linestyle="--")
#        axs.tick_params(
#            direction="in", left=True, right=True, top=True, bottom=True
#        )
#        axs.set_xlabel('Time [ns]')
#        axs.set_ylabel('Population')
#        plt.legend(model.state_labels)
#        plt.show()
#        #plt.savefig('test.pdf')
#
# plot_dynamics(exp,init_state, barely_a_seq)
#
#
