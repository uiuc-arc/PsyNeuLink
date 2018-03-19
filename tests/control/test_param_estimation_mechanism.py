import numpy as np

from psyneulink.components.functions.function import BogaczEtAl, Linear
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.components.system import System
from psyneulink.globals.keywords import ALLOCATION_SAMPLES, IDENTITY_MATRIX, MEAN, RESULT, VARIANCE
from psyneulink.library.mechanisms.processing.integrator.ddm import DDM, DECISION_VARIABLE, PROBABILITY_UPPER_THRESHOLD, \
    RESPONSE_TIME
from psyneulink.library.subsystems.param_estimator.paramestimationcontrolmechanism import \
    ParamEstimationControlMechanism

import hddm


# def test_hddm_randomness():
#
#     data = hddm.load_csv('tests/control/hddm_test_data.csv')
#     np.random.seed(1)
#     hddm_model = hddm.HDDM(data=data)
#     hddm_model.sample(6, burn=0, progress_bar=False)
#     print("")
#     print("Stochastics1: " + ', '.join([str(s) for s in hddm_model.mc.stochastics]))
#     hddm_v_gt = hddm_model.mc.trace('v')[:]
#
#     for i in range(30):
#
#         np.random.seed(1)
#         hddm_model2 = hddm.HDDM(data=data)
#         hddm_model2.sample(6, burn=0, progress_bar=False)
#         print("")
#         print("Stochastics2: " + ', '.join([str(s) for s in hddm_model2.mc.stochastics]))
#         hddm_v = hddm_model2.mc.trace('v')[:]
#
#         assert(np.alltrue(hddm_v_gt == hddm_v))

# def test_pecm_vs_hddm():
#
#     # Set the numpy random seed so we get predictable behaviour
#     np.random.seed(1)
#
#     # Create and HDDM model and sample from it to compare results.
#     hddm_model = hddm.HDDM(data=hddm.load_csv('tests/control/hddm_test_data.csv'))
#     hddm_model.sample(2, burn=0, progress_bar=False)
#     hddm_a, hddm_v = hddm_model.nodes_db.node[['a', 'v']]
#
#     print("")
#     for i in range(len(hddm_a.trace())):
#         print("HDDM Sampler: drift_rate={}, threshold={}".format(hddm_v.trace()[i], hddm_a.trace()[i]))
#
#     Decision = DDM(
#         function=BogaczEtAl(
#             drift_rate=(
#                 1.0,
#                 ControlProjection(
#                     function=Linear,
#                     control_signal_params={
#                         ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
#                     },
#                 ),
#             ),
#             threshold=(
#                 1.0,
#                 ControlProjection(
#                     function=Linear,
#                     control_signal_params={
#                         ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
#                     },
#                 ),
#             ),
#             noise=(0.5),
#             starting_point=(0),
#             t0=0.45
#         ),
#         output_states=[
#             DECISION_VARIABLE,
#             RESPONSE_TIME,
#             PROBABILITY_UPPER_THRESHOLD
#         ],
#         name='Decision',
#     )
#
#     # Processes:
#     TaskExecutionProcess = Process(
#         pathway=[Decision],
#         name='TaskExecutionProcess',
#     )
#
#     # System:
#     mySystem = System(
#         processes=[TaskExecutionProcess],
#         controller=ParamEstimationControlMechanism(data_in_file='tests/control/hddm_test_data.csv'),
#         enable_controller=True,
#         monitor_for_control=[
#             Decision.PROBABILITY_UPPER_THRESHOLD,
#             (Decision.RESPONSE_TIME, -1, 1)],
#         name='Param Estimation Test System',
#     )
#
#     # Stimuli
#     stim_list_dict = {
#         Decision: np.ones((1,))
#     }
#
#     # Set the numpy random seed so we get predictable behaviour
#     np.random.seed(1)
#
#     mySystem.run(inputs=stim_list_dict)

def test_pecm():

    Decision = DDM(
        function=BogaczEtAl(
            drift_rate=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal_params={
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            threshold=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal_params={
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            noise=(0.5),
            starting_point=(0),
            t0=0.45
        ),
        output_states=[
            DECISION_VARIABLE,
            RESPONSE_TIME,
            PROBABILITY_UPPER_THRESHOLD
        ],
        name='Decision',
    )

    # Processes:
    TaskExecutionProcess = Process(
        pathway=[Decision],
        name='TaskExecutionProcess',
    )

    # System:
    mySystem = System(
        processes=[TaskExecutionProcess],
        controller=ParamEstimationControlMechanism(data_in_file='tests/control/hddm_test_data.csv'),
        enable_controller=True,
        monitor_for_control=[
            Decision.PROBABILITY_UPPER_THRESHOLD,
            (Decision.RESPONSE_TIME, -1, 1)],
        name='Param Estimation Test System',
    )

    # TaskExecutionProcess.prefs.paramValidationPref = False
    # RewardProcess.prefs.paramValidationPref = False
    # mySystem.prefs.paramValidationPref = False

    # Stimuli
    stim_list_dict = {
        Decision: [1]
    }

    mySystem.run(inputs=stim_list_dict)