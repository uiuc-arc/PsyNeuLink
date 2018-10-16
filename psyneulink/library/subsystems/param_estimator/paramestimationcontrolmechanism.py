# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  ParamEstimationControlMechanism ******************************************************

"""

Overview
--------

A ParamEstimationControlMechanism is a `ControlMechanism <ControlMechanism>` that regulates it `ControlSignals <ControlSignal>` in order
to optimize the performance of the System to which it belongs. It does so by fitting an underlying statistical model to
data provided. Currently, it only supports one such statistical model and that is the hierarchical full drift diffusion
model as implemented by the HDDM python package.

Creating an ParamEstimationControlMechanism
------------------------------------------------

An ParamEstimationControlMechanism should be created using its constructor and passed as the controller argument to a
System.

.. note::
   Although a ParamEstimationControlMechanism should be created on its own, it can only be assigned to, and executed within a `System` as
   the System's `controller <System.controller>`.

When an ParamEstimationControlMechanism is assigned to, or created by a System, it is assigned the OutputStates to be monitored and
parameters to be controlled specified for that System (see `System_Control`).
"""

import numpy as np
import typecheck as tc

import hddm

from psyneulink import ContextFlags, EVCAuxiliaryError
from psyneulink.library.subsystems.param_estimator.hddm_psyneulink import HDDMPsyNeuLink

from psyneulink.globals.defaults import defaultControlAllocation
from psyneulink.components.functions.function import Function_Base
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.components.functions.function import LinearCombination
from psyneulink.globals.keywords import CONTROL, COST_FUNCTION, FUNCTION, INITIALIZING, \
    INIT_FUNCTION_METHOD_ONLY, PARAMETER_STATES, PREDICTION_MECHANISM, PREDICTION_MECHANISM_PARAMS, \
    PREDICTION_MECHANISM_TYPE, SUM, PARAM_EST_MECHANISM, COMBINE_OUTCOME_AND_COST_FUNCTION, COST_FUNCTION, \
    EXECUTING, FUNCTION_OUTPUT_TYPE_CONVERSION, INITIALIZING, PARAMETER_STATE_PARAMS, kwPreferenceSetName, \
    kwProgressBarChar, COMMAND_LINE
from psyneulink.components.shellclasses import Function, System_Base
from psyneulink.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.components.mechanisms.mechanism import MechanismList
from psyneulink.components.mechanisms.processing import integratormechanism
from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.scheduling.time import TimeScale



kwParamEstimationFunction = "PARAM ESTIMATION FUNCTION"
kwParamEstimationFunctionType = "PARAM ESTIMATION FUNCTION TYPE"
MCMC_PARAM_SAMPLE_FUNCTION = "MCMC PARAMETER SAMPLING FUNCTION"

class MCMCParamSampler(Function_Base):
    """
    A function that generates random samples of parameter values using MCMC sampling. Currently, it utilizes the
    underlying statistical model of the DDM implemented by the HDDM library. Each sample drawn from the HDDM model
    during parameter estimation is returned as an allocation policy.

    This is the default function assigned to the ParamEstimationControlMechanism
    """
    componentName = MCMC_PARAM_SAMPLE_FUNCTION

    class ClassDefaults(Function_Base.ClassDefaults):
        variable = None

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: False,
        PARAMETER_STATE_PARAMS: None})

    classPreferences = {
        kwPreferenceSetName: 'ValueFunctionCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    @tc.typecheck
    def __init__(self,
        default_variable=None,
        function=None,
        variable = None,
        params = None,
        owner = None,
        prefs: is_pref_set = None,
        context = None):

        function = function or self.function

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params)
        self.aux_function = function

        super().__init__(default_variable=variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR,
                         function=function)

        # This dictionary maps between how DDM parameters in PsyNeuLink are called and how HDDM references them.
        self.pnl_ddm_param_to_hddm = {'threshold': 'a', 'drift_rate': 'v', 'drift_rate_std': 'sv',
                                      'bias': 'z', 'bias_std': 'sz',
                                      'non_decision_time': 't', 'non_decision_time_std': 'st'}

    def function(
        self,
        controller=None,
        variable=None,
        runtime_params=None,
        time_scale=TimeScale.TRIAL,
        params=None,
        context=None,
    ):

        if (self.context.initialization_status == ContextFlags.INITIALIZING or
                self.owner.context.initialization_status == ContextFlags.INITIALIZING):
            return np.array([1.0], dtype=np.object)

        # Get value of, or set default for standard args
        if controller is None:
            raise EVCAuxiliaryError("Call to ControlSignalGridSearch() missing controller argument")

        # FIXME: This is a hack. We need to pass the current context to the likelihood before
        # it gets called during sampling. This is because the likelihood function will make
        # a call to run_simulation on the controller.
        controller.hddm_model.pnl_likelihood_estimator.context = context

        # Run the MCMC sampling
        self.owner.hddm_model.sample(1, burn=0, progress_bar=False)

        # Go through each control signal and grab the appropriate value for from the sample to create the allocation
        allocation_policy = np.zeros((len(self.owner.control_signals),1), dtype=np.object)
        for i in range(len(self.owner.control_signals)):
            # Get the receiving name of the parameter
            param_name = self.owner.control_signals[i].projections[0].receiver.name

            try:
                # Get the statistical model name for this parameter
                stat_model_param_name = self.pnl_ddm_param_to_hddm[param_name]
            except KeyError:
                #raise ValueError('Could not find appropriate HDDM parameter for Control Signal {}'.format(param_name))
                pass

            try:
                # Extract the value from the PyMC trace based on this name
                allocation_policy[i] = self.owner.hddm_model.nodes_db.node[stat_model_param_name].trace()[0]
            except KeyError:
                allocation_policy[i] = 0.0

        # Assign our allocation policy
        controller.allocation_policy = allocation_policy

        return controller.allocation_policy


class ParamEstimationControlMechanism(ControlMechanism):
    componentType = PARAM_EST_MECHANISM
    initMethod = INIT_FUNCTION_METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.SUBTYPE

    class ClassDefaults(ControlMechanism.ClassDefaults):
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = defaultControlAllocation
        function = MCMCParamSampler

    paramClassDefaults = ControlMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({PARAMETER_STATES: NotImplemented})  # This suppresses parameterStates

    @tc.typecheck
    def __init__(self,
                 data_in_file,
                 system: tc.optional(System_Base) = None,
                 control_signals: tc.optional(list) = None,
                 objective_mechanism: tc.optional(tc.any(ObjectiveMechanism, list)) = None,
                 function=MCMCParamSampler,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(system=system,
                                                  objective_mechanism=objective_mechanism,
                                                  function=function,
                                                  control_signals=control_signals,
                                                  params=params)

        # Create a statistical model using HDDM.
        self.data = hddm.load_csv(data_in_file)

        super(ParamEstimationControlMechanism, self).__init__(
            system=system,
            function=function,
            objective_mechanism=objective_mechanism,
            control_signals=control_signals,
            params=params,
            name=name,
            prefs=prefs)

    @tc.typecheck
    def assign_as_controller(self, system: System_Base, context=ContextFlags.COMMAND_LINE):
        super().assign_as_controller(system=system, context=context)
        self.hddm_model = HDDMPsyNeuLink(data=self.data, system=system)

    def _execute(
        self,
        variable=None,
        runtime_params=None,
        context=None
    ):
        allocation_policy = super(ControlMechanism, self)._execute(
            controller=self,
            variable=variable,
            runtime_params=runtime_params,
            context=context
        )
        return allocation_policy

    def run_simulation(self,
                       inputs,
                       allocation_vector,
                       termination_processing=None,
                       runtime_params=None,
                       context=None):
        """
        Run simulation of `System` for which the ParamEstimationControlMechanism is the `controller <System.controller>`.

        Arguments
        ----------

        inputs : List[input] or ndarray(input) : default default_variable
            the inputs used for each in a sequence of executions of the Mechanism in the `System`.  This should be the
            `value <Mechanism_Base.value> for each `prediction Mechanism <EVCControlMechanism_Prediction_Mechanisms>` listed
            in the `prediction_mechanisms` attribute.  The inputs are available from the `predicted_input` attribute.

        allocation_vector : (1D np.array)
            the allocation policy to use in running the simulation, with one allocation value for each of the
            EVCControlMechanism's ControlSignals (listed in `control_signals`).

        runtime_params : Optional[Dict[str, Dict[str, Dict[str, value]]]]
            a dictionary that can include any of the parameters used as arguments to instantiate the mechanisms,
            their functions, or Projection(s) to any of their states.  See `Mechanism_Runtime_Parameters` for a full
            description.

        """
        if self.value is None:
            # Initialize value if it is None
            self.value = self.allocation_policy

        # Implement the current allocation_policy over ControlSignals (outputStates),
        #    by assigning allocation values to EVCControlMechanism.value, and then calling _update_output_states
        allocation_policy = np.empty(len(allocation_vector), dtype=np.object)
        allocation_policy[:] = allocation_vector
        self.value = allocation_policy
        # for i in range(len(self.control_signals)):
        #     # self.control_signals[list(self.control_signals.values())[i]].value = np.atleast_1d(allocation_vector[i])
        #     self.value[i] = np.atleast_1d(allocation_vector[i])

        self._update_output_states(runtime_params=runtime_params, context=context)

        # Buffer System attributes
        execution_id_buffer = self.system._execution_id
        animate_buffer = self.system._animate

        self.system.context.execution_phase = ContextFlags.SIMULATION
        result = self.system.run(inputs=inputs, context=context, termination_processing=termination_processing)
        self.system.context.execution_phase = ContextFlags.IDLE

        # Restore System attributes
        self.system._animate = animate_buffer
        self.system._execution_id = execution_id_buffer

        # Get outcomes for current allocation_policy
        #    = the values of the monitored output states (self.input_states)
        # self.objective_mechanism.execute(context=CONTROL_SIMULATION)
        monitored_states = self._update_input_states(runtime_params=runtime_params, context=context)

        #for i in range(len(self.control_signals)):
        #    self.control_signal_costs[i] = self.control_signals[i].cost

        return result