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

from psyneulink.library.subsystems.param_estimator.hddm_psyneulink import HDDMPsyNeuLink

from psyneulink.globals.defaults import defaultControlAllocation
from psyneulink.components.functions.function import Function_Base
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref, kpRuntimeParamStickyAssignmentPref
from psyneulink.components.functions.function import LinearCombination
from psyneulink.globals.keywords import CONTROL, COST_FUNCTION, FUNCTION, INITIALIZING, \
    INIT_FUNCTION_METHOD_ONLY, PARAMETER_STATES, PREDICTION_MECHANISM, PREDICTION_MECHANISM_PARAMS, \
    PREDICTION_MECHANISM_TYPE, SUM, PARAM_EST_MECHANISM, COMBINE_OUTCOME_AND_COST_FUNCTION, COST_FUNCTION, \
    EXECUTING, FUNCTION_OUTPUT_TYPE_CONVERSION, INITIALIZING, PARAMETER_STATE_PARAMS, kwPreferenceSetName, kwProgressBarChar
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
    componentType = kwParamEstimationFunctionType

    class ClassDefaults(Function_Base.ClassDefaults):
        variable = None

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: False,
        PARAMETER_STATE_PARAMS: None})

    # MODIFIED 11/29/16 NEW:
    classPreferences = {
        kwPreferenceSetName: 'ValueFunctionCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    def __init__(self,
                 function=None,
                 variable=None,
                 default_variable=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params)
        self.aux_function = function

        super().__init__(default_variable=variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

    def function(
        self,
        controller=None,
        variable=None,
        runtime_params=None,
        time_scale=TimeScale.TRIAL,
        params=None,
        context=None,
    ):

        if INITIALIZING in context:
            return defaultControlAllocation

        # Run the MCMC sampling
        self.owner.hddm_model.sample(22, burn=20, progress_bar=False)

        # Get a sample from the trace for each parameter we are estimating.
        drift_rate = self.owner.hddm_model.mc.trace('v')[1]
        threshold = self.owner.hddm_model.mc.trace('a')[1]

        #print("MCMCParamSampler: drift_rate={}, threshold={}".format(drift_rate, threshold))

        # Assign our allocation policy
        controller.allocation_policy = np.array([[drift_rate, threshold]]).T

        return controller.allocation_policy


class ParamEstimationControlMechanism(ControlMechanism):
    componentType = PARAM_EST_MECHANISM
    initMethod = INIT_FUNCTION_METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.SUBTYPE

    class ClassDefaults(ControlMechanism.ClassDefaults):
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = defaultControlAllocation

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
        data = hddm.load_csv(data_in_file)

        self.hddm_model = HDDMPsyNeuLink(data)

        super(ParamEstimationControlMechanism, self).__init__(
            system=system,
            function=function,
            objective_mechanism=objective_mechanism,
            control_signals=control_signals,
            params=params,
            name=name,
            prefs=prefs,
            context=self)

    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    context=None):
        """Determine `allocation_policy <ParamEstimationControlMechanism.allocation_policy>` for next run of System

        Call self.function -- default is MCMCParamSearch
        Return an allocation_policy
        """


        # IMPLEMENTATION NOTE:
        # self.system._store_system_state()

        allocation_policy = self.function(controller=self,
                                          variable=variable,
                                          runtime_params=runtime_params,
                                          context=context)
        # IMPLEMENTATION NOTE:
        # self.system._restore_system_state()

        return allocation_policy