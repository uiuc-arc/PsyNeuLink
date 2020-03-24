#%%
import psyneulink as pnl
import numpy as np
import pytest
import time

# Load Bryant's data
#import pandas as pd
#data = pd.read_csv('rtdist/example/DataTaskSwitchingRecoded.csv')
#data = data[data.subjectID == 1]

# BEGIN: Composition Construction

# Parameters to be optimized
GAIN = 1.0          # Gain parameter of the task (activation) mechanism
THRESHOLD = 1.0     # Threshold parameter of the LCA (decisionMaker) mechanism

# Parameters that are fixed (for now)
tau = 0.9           # Rate parameter of task mechanism's adaptive integrator
AUTOMATICITY = 0.2  # Contribution of automatic processing to evidence accumulation for each response
LEAK = 0.2          # Leak parameter of the LCA
COMPETITION = 1.0   # Competition parameter of the LCA
NOISE = 0.04        # Noise parameter of the LCA
#rng = np.random.RandomState()
#NOISE = lambda: rng.normal(scale=NOISE)
TIME_STEP_SIZE = 0.01

# Task Layer: [Color, Shape] {0, 1} Mutually Exclusive
# Origin Node
taskLayer = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                  size=2,
                                  function=pnl.Linear(slope=1, intercept=0),
                                  output_ports=[pnl.RESULT],
                                  name='Task Input [I1, I2]')

# Stimulus Layer: [Blue, Green, Heart, Diamond]
# Origin Node
stimulusInfo = pnl.TransferMechanism(default_variable=[[0.0, 0.0, 0.0, 0.0]],
                                     size=4,
                                     function=pnl.Linear(slope=1, intercept=0),
                                     output_ports=[pnl.RESULT],
                                     name='Stimulus Input [S1, S2, S3, S4]')

# Activation Layer: [Color Activation, Shape Activation]
# Recurrent: Self Excitation, Mutual Inhibition
# Controlled: Gain Parameter
activation = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0]],
                                            function=pnl.Logistic(gain=GAIN),
                                            matrix=[[1.0, -1.0],
                                                    [-1.0, 1.0]],
                                            integrator_mode=True,
                                            integrator_function=pnl.AdaptiveIntegrator(rate=(tau)),
                                            initial_value=np.array([[0.0, 0.0]]),
                                            output_ports=[pnl.RESULT],
                                            name='Task Activations [Act1, Act2]')

# Hadamard product of Activation and Stimulus Information
nonAutomaticComponent = pnl.TransferMechanism(default_variable=[[0.0, 0.0, 0.0, 0.0]],
                                              size=4,
                                              function=pnl.Linear(slope=1, intercept=0),
                                              input_ports=pnl.InputPort(combine=pnl.PRODUCT),
                                              output_ports=[pnl.RESULT],
                                              name='Non-Automatic Component [S1*Act1, S2*Act2, S3*Act3, S4*Act4]')

# Mapping Projection from activation to nonAutomaticComponent
# Changes the activation [Color, Shape] to [Color, Color, Shape, Shape]
activation_to_nonAutomaticComponent = pnl.MappingProjection(sender=activation,
                                                            receiver=nonAutomaticComponent,
                                                            matrix=np.asarray([[1, 1, 0, 0],
                                                                               [0, 0, 1, 1]]))

# Summation of nonAutomatic and Automatic Components
lcaCombination = pnl.TransferMechanism(default_variable=[[0.0, 0.0, 0.0, 0.0]],
                                       size=4,
                                       function=pnl.Linear(slope=1, intercept=0),
                                       input_ports=pnl.InputPort(combine=pnl.SUM),
                                       output_ports=[pnl.RESULT],
                                       name='Response Act = [S1 + S1*Act1, S2 + S2*Act2, S3 + S3*Act3, S4 + S4*Act4]')

# Mapping Projection from stimulusInfo to lcaCombination
# Multiplies stimulus input by the automaticity weight
stimulusInfo_to_lcaCombination = pnl.MappingProjection(sender=stimulusInfo,
                                                       receiver=lcaCombination,
                                                       matrix=np.asarray([[AUTOMATICITY, 0, 0, 0],
                                                                          [0, AUTOMATICITY, 0, 0],
                                                                          [0, 0, AUTOMATICITY, 0],
                                                                          [0, 0, 0, AUTOMATICITY]]))

# Currently necessary to return the index of the accumulator that reached threshold
def returnIndexOfMaxValue(x):
    return [list(x[0]).index(max(list(x[0])))]

# Compute accuracy of the LCA
def computeAccuracy(owner_value, taskInput, stimulusInput):
    owner_value, taskInput, stimulusInput = list(owner_value), list(taskInput), list(stimulusInput)

    colorTrial = (taskInput[0] > 0)
    shapeTrial = (taskInput[1] > 0)

    if colorTrial:
        if stimulusInput[0] > stimulusInput[1]:
            CorrectResp = 0
        if stimulusInput[1] > stimulusInput[0]:
            CorrectResp = 1
    if shapeTrial:
        if stimulusInput[2] > stimulusInput[3]:
            CorrectResp = 2
        if stimulusInput[3] > stimulusInput[2]:
            CorrectResp = 3

    Resp = owner_value.index(max(owner_value))

    if CorrectResp == Resp:
        Accuracy = 1
    if CorrectResp != Resp:
        Accuracy = 0
    return Accuracy

decisionMaker = pnl.LCAMechanism(default_variable=[[0.0, 0.0, 0.0, 0.0]],
                                 size=4,
                                 function=pnl.Linear(slope=1, intercept=0),
                                 reinitialize_when=pnl.AtTrialStart(),
                                 leak=LEAK,
                                 competition=COMPETITION,
                                 self_excitation=0,
                                 noise=NOISE,
                                 threshold=THRESHOLD,
                                 output_ports=[pnl.RESULT,
                                               {pnl.NAME: "EXECUTION COUNT",
                                                pnl.VARIABLE: pnl.OWNER_EXECUTION_COUNT},
                                               {pnl.NAME: "RESPONSE",
#                                                pnl.VARIABLE: pnl.OWNER_VALUE,
                                                pnl.FUNCTION: pnl.OneHot(mode=pnl.MAX_INDICATOR)}
                                                ],
                                 time_step_size=TIME_STEP_SIZE,
                                 clip=[0.0, THRESHOLD])

# Currently necessary to manually reset the execution count of the LCA for each trial
# Call this in run using call_after_trial
def reset_lca_count():
  decisionMaker.execution_count=0

taskLayer.set_log_conditions([pnl.RESULT])
stimulusInfo.set_log_conditions([pnl.RESULT])
activation.set_log_conditions([pnl.RESULT, "mod_gain"])
nonAutomaticComponent.set_log_conditions([pnl.RESULT])
lcaCombination.set_log_conditions([pnl.RESULT])
decisionMaker.set_log_conditions([pnl.RESULT, pnl.VALUE])


# Composition Creation

stabilityFlexibility = pnl.Composition(controller_mode=pnl.BEFORE)

# Node Creation
stabilityFlexibility.add_node(taskLayer)
stabilityFlexibility.add_node(activation)
stabilityFlexibility.add_node(nonAutomaticComponent)
stabilityFlexibility.add_node(stimulusInfo)
stabilityFlexibility.add_node(lcaCombination)
#stabilityFlexibility.add_node(decisionMaker, required_roles=pnl.NodeRole.OUTPUT)

# Projection Creation
stabilityFlexibility.add_projection(sender=taskLayer, receiver=activation)
stabilityFlexibility.add_projection(activation_to_nonAutomaticComponent)
stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=nonAutomaticComponent)
stabilityFlexibility.add_projection(stimulusInfo_to_lcaCombination)
stabilityFlexibility.add_projection(sender=nonAutomaticComponent, receiver=lcaCombination)
#stabilityFlexibility.add_projection(sender=lcaCombination, receiver=decisionMaker)

# Schedule execution of mechanisms
# This is necessary to ensure everything but the LCA executes only once per trial, i.e. at the first pass
#stabilityFlexibility.scheduler.add_condition(taskLayer, pnl.AtPass(0))
#stabilityFlexibility.scheduler.add_condition(stimulusInfo, pnl.AtPass(0))
#stabilityFlexibility.scheduler.add_condition(decisionMaker, pnl.Always())
def task_onehot(task):
    return {'color': [1, 0],
            'shape': [0, 1]}[task]
#taskTrain = [task_onehot(x) for x in data.task_cued.values]

# Origin Node Inputs
taskTrain = [[1, 0],    # Color trial
             [1, 0],    # Color trial, repetition
             [0, 1],    # Shape trial, switch
             [0, 1],    # Shape trial, repetition
             [1, 0],    # Color trial, switch
             [1, 0],    # Color trial, repetition
             [0, 1],    # Shape trial, switch
             [0, 1]]    # Shape trial, repetition

def stim_encode(stim):
    return {'blue_heart':    [1, 0, 1, 0],
            'green_diamond': [0, 1, 0, 1],
            'green_heart':   [0, 1, 1, 0],
            'blue_diamond':  [1, 0, 0, 1]
            }[stim]
#stimulusTrain = [stim_encode(x) for x in data.stimulus_identity.values]

stimulusTrain = [[1, 0, 1, 0],  # Blue heart
                 [0, 1, 0, 1],  # Green diamond
                 [0, 1, 1, 0],  # Green heart
                 [1, 0, 0, 1],  # Blue diamond
                 [1, 0, 1, 0],  # Blue heart
                 [0, 1, 0, 1],  # Green diamond
                 [0, 1, 1, 0],  # Green heart
                 [1, 0, 0, 1]]  # Blue diamond

inputs = {taskLayer: taskTrain, stimulusInfo: stimulusTrain}

# taskLayer.reportOutputPref=True
# stimulusInfo.reportOutputPref=True
# activation.reportOutputPref=True
# nonAutomaticComponent.reportOutputPref=True
# lcaCombination.reportOutputPref=True
# decisionMaker.reportOutputPref=True

N_SAMPLES = 1

t1 = time.time()
MODE="LLVMRun" # or Python
stabilityFlexibility.run(inputs,
                         call_after_trial=reset_lca_count, bin_execute=MODE,
                         num_trials=N_SAMPLES*len(taskTrain))
t2 = time.time()
stabilityFlexibility.run(inputs,
                         call_after_trial=reset_lca_count, bin_execute=MODE,
                         num_trials=N_SAMPLES*len(taskTrain))
t3 = time.time()
print("Time to run in", MODE, "mode:", t2 - t1)
print("Time to run in", MODE, "mode:", t3 - t2)


#taskLayer.log.print_entries(),
#stimulusInfo.log.print_entries(),
#activation.log.print_entries(),
#nonAutomaticComponent.log.print_entries(),
#lcaCombination.log.print_entries(),
#decisionMaker.log.print_entries(),
def _exec_count(i, c):
    if i == 0 or MODE == "Python":
        return c
    return c[0] - stabilityFlexibility.results[i-1][1][0]
results = ([r[0], [_exec_count(i, r[1])], list(r[2]).index(1.0)] for i, r in enumerate(stabilityFlexibility.results))

lcaV = np.transpose(np.array([x[0] for x in stabilityFlexibility.results[0:len(taskTrain)]]))
np.save('Scripts/Debug/lcaV.npy', lcaV)
print(f"lcaV: mean={np.mean(lcaV)}, std={np.std(lcaV)} min={np.min(lcaV)}, max={np.max(lcaV)}")

# f, axs = plt.subplots(4,1, sharey=True)
# for i in range(4):
#     axs[i].hist(lcaV[i, lcaV[i, :] != 0.0], bins=50, range=(np.min(lcaV), np.max(lcaV)))

# for trial in results:
#     print(trial)

# results = list(results)
# rts = np.array([r[1][0] for r in results])
# rts[0] = rts[0][0]
# responses = np.array([r[2] for r in results])
#
# import matplotlib.pyplot as plt
# plt.ion()
# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# axs[0].hist(rts, bins=25)
# axs[1].hist(rts, bins=25)
