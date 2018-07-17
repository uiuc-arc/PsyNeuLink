import numpy as np
from hddm import wfpt

from psyneulink import ContextFlags


class SystemLikelihoodEstimator:
    """
    The SystemLikelihoodEstimator class provides support for estimating the log probability of a systems output
    conditioned on its current parameter states values. Essentially, it provides a common interface for computing this
    log probability by exposing a method which computes this value.
    """
    def __init__(self, system):
        self.system = system

    def get_likelihood_function(self, **kwargs):
        """
            A simple function that returns another function object that calls the HDDM Navarro and Fuss
            Cython implementation from HDDM usings a specified set of wiener parameters.
            :param wp: The wiener parameters to provide the the likelihood function. See HDDM documentation.
            :return: A function implementing the likelihood.
            """

        def wfpt_like(x, v, sv, a, z, sz, t, st, p_outlier=0):

            if self.system.controller.context.execution_phase == ContextFlags.SIMULATION:
                # Run simulations of the PsyNeuLink system, we will use the outputs of these simulations to estimate the
                # conditional log probability
                input = {self.system.origin_mechanisms[0] : [1]}

                control_signals = self.system.controller.control_signals

                # Search through control signals and find appropriate psyneulink parameter in the system
                # to map this value to.
                allocation_values = []
                for i in range(len(control_signals)):

                    # Get the receiving name of the parameter control signal
                    param_name = control_signals[i].projections[0].receiver.name

                    if param_name is "drift_rate":
                        allocation_values.append(v)
                    elif param_name is "drift_rate_std":
                        allocation_values.append(sv)
                    elif param_name is "bias":
                        allocation_values.append(z)
                    elif param_name is "bias_std":
                        allocation_values.append(sz)
                    elif param_name is "non_decision_time":
                        allocation_values.append(t)
                    elif param_name is "non_decision_time_std":
                        allocation_values.append(st)
                    elif param_name is "threshold":
                        allocation_values.append(a)
                    elif param_name is "response_time":
                        allocation_values.append(x['rt'].values)

                result = self.system.controller.run_simulation(inputs=input, allocation_vector=allocation_values)

            if np.all(~np.isnan(x['rt'])):
                return wfpt.wiener_like(x['rt'].values, v, sv, a, z, sz, t, st,
                                        p_outlier=p_outlier, **kwargs)
            else:  # for missing RTs. Currently undocumented.
                noresponse = np.isnan(x['rt'])
                ## get sum of log p for trials with RTs as usual ##
                LLH_resp = wfpt.wiener_like(x.loc[-noresponse, 'rt'].values,
                                            v, sv, a, z, sz, t, st, p_outlier=p_outlier, **kwargs)

                ## get sum of log p for no-response trials from p(upper_boundary|parameters) ##
                # this function assumes following format for the RTs:
                # - accuracy coding such that correct responses have a 1 and incorrect responses a 0
                # - usage of HDDMStimCoding for z
                # - missing RTs are coded as 999/-999
                # - note that hddm will flip RTs, such that error trials have negative RTs
                # so that the miss-trial in the go condition and comission error
                # in the no-go condition will have negative RTs

                # get number of no-response trials
                n_noresponse = sum(noresponse)

                # percentage correct according to probability to get to upper boundary
                if v == 0:
                    p_correct = z
                else:
                    p_correct = (np.exp(-2 * a * z * v) - 1) / (np.exp(-2 * a * v) - 1)

                # calculate percent no-response trials from % correct
                if sum(x.loc[noresponse, 'rt']) > 0:
                    p_noresponse = p_correct  # when no-response trials have a positive RT
                    # we are looking at nogo Trials
                else:
                    p_noresponse = 1 - p_correct  # when no-response trials have a
                    # negative RT we are looking at go Trials

                # likelihood for no-response trials
                LLH_noresp = np.log(p_noresponse) * n_noresponse

                return LLH_resp + LLH_noresp

        return wfpt_like