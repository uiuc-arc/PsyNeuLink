# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  PredictionProjection ***********************************************

"""
.. _Prediction_Overview:

Overview
--------

An PredictionProjection is a subclass of `MappingProjection` that connects prediction mechanisms to origin mechanisms
during simulations only.

.. _Prediction_Creation:

Creating a PredictionProjection
-------------------------------------

A PredictionProjection is created automatically when a prediction mechanism is instantiated. It is not recommended to
create a PredictionProjection on its own, because the PredictionProjection is executed at specific times (namely, only
during control simulations), and may cause unexpected behavior outside of its intended use case.

.. _Prediction_Structure:

Prediction Projection Structure
--------------------------

In structure, the PredictionProjection is identical to a MappingProjection. The only difference is that the
PredictionProjection is ignored during normal execution.

.. _Prediction_Configurable_Attributes:

Configurable Attributes
~~~~~~~~~~~~~~~~~~~~~~~

The only configurable parameter is the matrix, configured through the **matrix** argument.

.. _Prediction_Matrix:

* **matrix** - multiplied by the input to the PredictionProjection in order to produce the output. Specification of
  the **matrix** arguments determines the values of the matrix.

.. _Prediction_Execution:

Execution
---------

A PredictionProjection uses its `matrix <PredictionProjection.matrix>` parameter to transform the value of its
`sender <PredictionProjection.sender>`, and provide the result as input for its
`receiver <PredictionProjection.receiver>`.

The PredictionProjection is always ignored during standard execution. If learning is enabled over the prediction
process, then its `matrix <PredictionProjection.matrix>` updates during the learning phase. The PredictionProjection
only executes during control simulations.

.. _Prediction_Class_Reference:

Class Reference
---------------

"""
import numbers

import numpy as np
import typecheck as tc

from psyneulink.components.component import parameter_keywords
from psyneulink.components.functions.function import get_matrix
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.projections.projection import projection_keywords
from psyneulink.components.shellclasses import Mechanism
from psyneulink.components.states.outputstate import OutputState
from psyneulink.globals.keywords import PREDICTION_PROJECTION, DEFAULT_MATRIX, HOLLOW_MATRIX, MATRIX
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.globals.keywords import EVC_SIMULATION
__all__ = [
    'PredictionProjectionError', 'PredictionProjection', 'get_auto_matrix', 'get_hetero_matrix',
]

parameter_keywords.update({PREDICTION_PROJECTION})
projection_keywords.update({PREDICTION_PROJECTION})

class PredictionProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

class PredictionProjection(MappingProjection):
    """
    PredictionProjection(                              \
        owner=None,                                         \
        sender=None,                                        \
        receiver=None,                                      \
        matrix=DEFAULT_MATRIX,                              \
        params=None,                                        \
        name=None,                                          \
        prefs=None                                          \
        context=None)

    Implements a MappingProjection that connects prediction mechanisms to their corresponding origin mechanisms during
    control simulations only.

    Arguments
    ---------

    owner : Optional[Mechanism]
        simply specifies both the sender and receiver of the PredictionProjection. Setting owner=myMechanism is
        identical to setting sender=myMechanism and receiver=myMechanism.

    sender : Optional[OutputState or Mechanism]
        specifies the source of the Projection's input. If a Mechanism is specified, its
        `primary OutputState <OutputState_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used.

    receiver : Optional[InputState or Mechanism]
        specifies the destination of the Projection's output.  If a Mechanism is specified, its
        `primary InputState <InputState_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used.

    matrix : list, np.ndarray, np.matrix, function or keyword : default DEFAULT_MATRIX
        the matrix used by `function <PredictionProjection.function>` (default: `LinearCombination`) to transform
        the value of the `sender <PredictionProjection.sender>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the Projection's default assignment (`LinearCombination`).  Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default PredictionProjection-<index>
        a string used for the name of the PredictionProjection. When a PredictionProjection is created by a
        RecurrentTransferMechanism, its name is assigned "<name of RecurrentTransferMechanism> recurrent projection"
        (see `Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Projection_Base.classPreferences]
        the `PreferenceSet` for the MappingProjection; if it is not specified, a default is assigned using
        `classPreferences` defined in __init__.py (see `PreferenceSet <LINK>` for details).

    Attributes
    ----------

    componentType : PREDICTION_PROJECTION

    sender : OutputState
        identifies the source of the Projection's input.

    receiver: InputState
        identifies the destination of the Projection.

    learning_mechanism : LearningMechanism
        source of error signal for that determine changes to the `matrix <PredictionProjection.matrix>` when
        `learning <LearningProjection>` is used.

    matrix : 2d np.ndarray
        matrix used by `function <PredictionProjection.function>` to transform input from the `sender
        <MappingProjection.sender>` to the value provided to the `receiver <PredictionProjection.receiver>`.

    has_learning_projection : bool : False
        identifies whether the PredictionProjection's `MATRIX` `ParameterState <ParameterState>` has been assigned
        a `LearningProjection`.

    value : np.ndarray
        Output of PredictionProjection, transmitted to `variable <InputState.variable>` of `receiver`.

    name : str
        a string used for the name of the PredictionProjection (see `Registry <LINK>` for conventions used in
        naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection_Base.classPreferences
        the `PreferenceSet` for PredictionProjection (see :doc:`PreferenceSet <LINK>` for details).
    """

    componentType = PREDICTION_PROJECTION
    className = componentType
    suffix = " " + className

    class ClassDefaults(MappingProjection.ClassDefaults):
        variable = np.array([[0]])    # function is always LinearMatrix that requires 1D input

    classPreferenceLevel = PreferenceLevel.TYPE

    # necessary?
    paramClassDefaults = MappingProjection.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 sender=None,
                 receiver=None,
                 matrix=DEFAULT_MATRIX,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=None):

        if owner is not None:
            if not isinstance(owner, Mechanism):
                raise PredictionProjectionError('Owner of Prediction Mechanism must either be None or a Mechanism')
            if sender is None:
                sender = owner
            if receiver is None:
                receiver = owner

        params = self._assign_args_to_param_dicts(function_params={MATRIX: matrix}, params=params)

        super().__init__(sender=sender,
                         receiver=receiver,
                         matrix=matrix,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _execute(self, variable, runtime_params=None, context=None):
        if EVC_SIMULATION not in context:
            self.sender.value *= 0.0

        return super()._execute(variable, runtime_params=runtime_params, context=context)