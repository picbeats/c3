"""Object that deals with the sensitivity test."""

import os
import json
import pickle
import itertools
import time
import numpy as np
import adaptive
import tensorflow as tf
import c3po.utils.display as display
from c3po.optimizers.optimizer import Optimizer
import matplotlib.pyplot as plt
from c3po.utils.utils import log_setup
from c3po.libraries.estimators import estimators


class SET():
    """Object that deals with the sensitivity test."""

    def __init__(
        self,
        dir_path,
        estimator_list,
        fom,
        sampling,
        batch_sizes,
        opt_map,
        state_labels=None,
        sweep_map=None,
        probe_list=[],
        accuracy_goal = 0.5,
        callback_foms=[],
        callback_figs=[],
        # algorithm_no_grad=None,
        # algorithm_with_grad=None,
        options={}
    ):
        """Initiliase."""
        # not really needed? it's not an optimization
        self.optim_status = {}

        self.estimator_list = estimator_list
        self.fom = fom
        self.sampling = sampling
        self.batch_sizes = batch_sizes
        self.opt_map = opt_map
        self.state_labels = state_labels
        self.sweep_map = sweep_map
        self.probe_list = probe_list
        self.accuracy_goal = accuracy_goal
        self.callback_foms = callback_foms
        self.callback_figs = callback_figs
        self.inverse = False
        self.options = options
        self.learn_data = {}
        self.log_setup(dir_path)

    def log_setup(self, dir_path):
        self.dir_path = os.path.abspath(dir_path)
        self.string = "set_test"
        # self.string = self.algorithm.__name__ + '-' \
        #          + self.sampling + '-' \
        #          + str(self.batch_size) + '-' \
        #          + self.fom.__name__
        # datafile = os.path.basename(self.datafile)
        # datafile = datafile.split('.')[0]
        # string = string + '----[' + datafile + ']'
        self.logdir = log_setup(dir_path, self.string)
        self.logname = 'model_learn.log'

    def start_log(self):
        self.start_time = time.time()
        start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write("Starting optimization at ")
            logfile.write(start_time_str)
            logfile.write("Optimization parameters:\n")
            logfile.write(json.dumps(self.opt_map))
            logfile.write("\n")
            logfile.flush()

    def set_exp(self, exp):
        self.exp = exp

    def read_data(self, datafiles):
        for target, datafile in datafiles.items():
                    with open(datafile, 'rb+') as file:
                        self.learn_data[target] = pickle.load(file)


    def load_best(self, init_point):
        with open(init_point) as init_file:
            best = init_file.readlines()
            best_exp_opt_map = [tuple(a) for a in json.loads(best[0])]
            init_p = json.loads(best[1])['params']
            self.exp.set_parameters(init_p, best_exp_opt_map)


    def select_from_data(self, batch_size):
        # TODO fix when batch size is 1 (atm it does all)
        learn_from = self.learn_from
        sampling = self.sampling
        indeces =  sampling(learn_from, batch_size)
        if self.inverse:
            return list(set(all) - set(indeces))
        else:
            return indeces


    def goal_run(self, val):
        exp_values = []
        exp_stds = []
        sim_values = []
        exp_shots = []

        #self.exp.set_parameters(current_params, self.opt_map, scaled=False)
        # print("tup: " + str(tup))
        # print("val: " + str(val))
        self.exp.set_parameters([val],[self.tup], scaled=False)
        print("params>>> ")
        params = self.exp.print_parameters(self.opt_map)
        print(params)

        print("self.learn_data.items(): " + str(len(self.learn_data.items())))
        count = 0
        for target, data in self.learn_data.items():

            self.learn_from = data['seqs_grouped_by_param_set']
            self.gateset_opt_map = data['opt_map']
            indeces = self.select_from_data(self.batch_sizes[target])

            print("indeces: " + str(len(indeces)))

            for ipar in indeces:
                if count % 100 == 0:
                        print("count: " + str(count))

                count += 1
                m = self.learn_from[ipar]
                gateset_params = m['params']
                gateset_opt_map = self.gateset_opt_map
                m_vals = m['results']
                m_stds = m['results_std']
                m_shots = m['shots']
                sequences = m['seqs']
                num_seqs = len(sequences)

                self.exp.gateset.set_parameters(
                    gateset_params, gateset_opt_map, scaled=False
                )

                # We find the unique gates used in the sequence and compute
                # only them.
                self.exp.opt_gates = list(
                    set(itertools.chain.from_iterable(sequences))
                )
                self.exp.get_gates()
                sim_vals = self.exp.evaluate(
                    sequences, labels=self.state_labels[target]
                )

                # exp_values.extend(m_vals)
                # exp_stds.extend(m_stds)
                sim_values.extend(sim_vals)


                with open(self.logdir + self.logname, 'a') as logfile:
                    logfile.write(
                        "\n  Parameterset {}, #{} of {}:\n {}\n {}\n".format(
                            ipar + 1,
                            count,
                            len(indeces),
                            json.dumps(self.gateset_opt_map),
                            self.exp.gateset.get_parameters(
                                self.gateset_opt_map, to_str=True
                            ),
                        )
                    )
                    logfile.write(
                        "Sequence    Simulation  Experiment  Std         "
                        "Diff\n"
                    )

                for iseq in range(num_seqs):
                    m_val = np.array(m_vals[iseq])
                    m_std = np.array(m_stds[iseq])
                    shots = np.array(m_shots[iseq])
                    exp_values.append(m_val)
                    exp_stds.append(m_std)
                    exp_shots.append(shots)
                    sim_val = sim_vals[iseq].numpy()
                    int_len = len(str(num_seqs))
                    with open(self.logdir + self.logname, 'a') as logfile:
                        for ii in range(len(sim_val)):
                            logfile.write(
                                f"{iseq + 1:8}    "
                                f"{float(sim_val[ii]):8.6f}    "
                                f"{float(m_val[ii]):8.6f}    "
                                f"{float(m_std[ii]):8.6f}    "
                                f"{float(shots[0]):8}    "
                                f"{float(m_val[ii]-sim_val[ii]):8.6f}\n"
                            )
                        logfile.flush()

        exp_values = tf.constant(exp_values, dtype=tf.float64)
        sim_values =  tf.stack(sim_values)
        if exp_values.shape != sim_values.shape:
            print(
                "C3:WARNING:"
                "Data format of experiment and simulation figures of"
                " merit does not match."
            )
        exp_stds = tf.constant(exp_stds, dtype=tf.float64)
#        print("exp_shots: " + str(exp_shots))
        exp_shots = tf.constant(exp_shots, dtype=tf.float64)
        goal = self.fom(exp_values, sim_values, exp_stds, exp_shots)
        goal_numpy = float(goal.numpy())

        with open(self.logdir + self.dfname, 'a') as datafile:
            datafile.write(f"{val}\t{goal_numpy}\n")

        for estimator in self.estimator_list:
            fom = estimators[estimator]
            tmp = fom(exp_values, sim_values, exp_stds, exp_shots)
            tmp = float(tmp.numpy())
            fname = estimator + '.dat'
            with open(self.logdir + fname, 'a') as datafile:
                datafile.write(f"{val}\t{tmp}\n")


        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write("\nFinished batch with ")
            logfile.write("{}: {}\n".format(self.fom.__name__, goal_numpy))
            print("{}: {}".format(self.fom.__name__, goal_numpy))
            for cb_fom in self.callback_foms:
                val = float(
                    cb_fom(exp_values, sim_values, exp_stds, exp_shots).numpy()
                )
                logfile.write("{}: {}\n".format(cb_fom.__name__, val))
                print("{}: {}".format(cb_fom.__name__, val))
            print("")
            logfile.flush()


#         for cb_fig in self.callback_figs:
            # fig = cb_fig(exp_values, sim_values.numpy()[0], exp_stds)
            # fig.savefig(
                # self.logdir
                # + cb_fig.__name__ + '/'
                # + 'eval:' + str(self.evaluation) + "__"
                # + self.fom.__name__ + str(round(goal_numpy, 3))
                # + '.png'
            # )
            # plt.close(fig)

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = goal_numpy
        self.evaluation += 1
        return goal_numpy





    def sensitivity_test(self):
        self.evaluation = 0

        self.logname = 'confirm_runner' + '.log'
        self.dfname = "data.dat"
        self.start_log()

        #for tmp in self.sweep_map:
        tmp = self.sweep_map[0]
        print(tmp)

        bound_min = tmp[2][0]
        bound_max = tmp[2][1]
        self.tup = (tmp[0],tmp[1])

        print(f"\nSaving as:\n{os.path.abspath(self.logdir + self.logname)}")

#         tmp = min(self.probe_list)
        # for i in range(30):
                # tmp = 0.9 * tmp
                # self.probe_list.append(tmp)

        probe_list_min = min(self.probe_list)
        probe_list_max = max(self.probe_list)

        bound_min = min(bound_min, probe_list_min)
        bound_max = max(bound_max, probe_list_max)

        print(" ")
        print("bound_min: " + str((bound_min)/(2e9 * np.pi)))
        print("bound_max: " + str((bound_max)/(2e9 * np.pi)))
        print(" ")

        learner = adaptive.Learner1D(self.goal_run, bounds=(bound_min, bound_max))

        if self.probe_list:
            for x in self.probe_list:
                print("from probe_list: " + str(x))
                tmp = learner.function(x)
                print("done\n")
                learner.tell(x, tmp)

        print("accuracy_goal: " + str(self.accuracy_goal))

        runner = adaptive.runner.simple(learner, goal=lambda learner_: learner_.loss() < self.accuracy_goal)


        # #=== Get the resulting data ======================================

        # Xs=np.array(list(learner.data.keys()))
        # Ys=np.array(list(learner.data.values()))
        # Ks=np.argsort(Xs)
        # Xs=Xs[Ks]
        # Ys=Ys[Ks]
