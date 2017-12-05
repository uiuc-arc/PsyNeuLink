"""
This implements a model by Nieuwenhuis et al. (2005) on "The Role of the Locus Coeruleus in Mediating the Attentional Blink:
A Neurocomputational Theory". The attentional blink refers to the temporary impairment in perceiving the 2nd of 2 targets presented in close temporal proximity.
<https://research.vu.nl/ws/files/2063874/Nieuwenhuis%20Journal%20of%20Experimental%20Psychology%20-%20General%20134(3)-2005%20u.pdf`.


# The aim is to reproduce Figure 3 from Nieuwenhuis et al. (2005) with Lag 2.


"""
import sys
import numpy as np

from psyneulink.library.mechanisms.processing.transfer.lca import LCA
from psyneulink.components.functions.function import Linear, Logistic, NormalDist
from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.system import System
from psyneulink.library.subsystems.agt.lccontrolmechanism import LCControlMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.globals.keywords import PROJECTION_TYPE, RECEIVER, SENDER, MATRIX

# --------------------------------- Global Variables ----------------------------------------

# First, we set the global variables, weights and initial values.
C = 0.90
initial_hv = 0.07                     # Initial value for h(v)
initial_w = 0.14                      # initial value u
d = 0.5                               # Uncorrelated Activity
initial_v = (initial_hv - (1-C)*d)/C  # get initial v from initial h(v)

SD = 0.15       # noise determined by standard deviation (SD)
a = 0.50        # Parameter describing shape of the FitzHugh–Nagumo cubic nullcline for the fast excitation variable v
k = 1.5         # Scaling factor for transforming NE release (u ) to gain (g ) on potentiated units
G = 0.5         # Base level of gain applied to decision and response units
dt = 0.02       # time step size

# Weights:
inpwt=1.5       # inpwt (Input to decision layer)
crswt=1/3       # crswt (Crosstalk input to decision layer)
inhwt=1.0       # inhwt (Mutual inhibition among decision units)
respinhwt=0     # respinhwt (Mutual inhibition among response units)  !!! WATCH OUT: this parameter is not mentioned in the original paper, most likely since it was set 0
decwt=3.5       # decwt (Target decision unit to response unit)
selfdwt=2.5     # selfdwt (Self recurrent conn. for each decision unit)
selfrwt=2.0     # selfrwt (Self recurrent conn. for response unit)
lcwt=0.3        # lcwt    (Target decision unit to LC)
decbias=1.75    # decbias (Bias input to decision units)
respbias=1.75   # respbias (Bias input to response units)
tau_v = 0.05    # Time constant for fast LC excitation variable v | NOTE: tau_v is misstated in the Gilzenrat paper(0.5)
tau_u = 5.00    # Time constant for slow LC recovery variable (‘NE release’) u
trials = 1100   # number of trials to reproduce Figure 3 from Nieuwenhuis et al. (2005)

# Create mechanisms ---------------------------------------------------------------------------------------------------

# First, we create the 3 layers of the behavioral network, i.e. INPUT LAYER, DECISION LAYER, and RESPONSE LAYER.

# Input Layer --- [ Target 1, Target 2, Distractor ]
input_layer = TransferMechanism(size = 3,                           # Number of units in input layer
                                initial_value= [[0.0,0.0,0.0]],     # Initial input values
                                name='INPUT LAYER')

# Create Decision Layer  --- [ Target 1, Target 2, Distractor ]
decision_layer = LCA(size=3,                                        # Number of units in input layer
                     initial_value= [[0.0,0.0,0.0]],                # Initial input values
                     time_step_size=dt,                             # Integration step size
                     leak=-1.0,                                     # Sets off diagonals to negative values
                     self_excitation=selfdwt,                       # Set diagonals to self excitate
                     competition=inhwt,                             # Set off diagonals to inhibit
                     #Kristin: Why do I need to specify gain here? Default should be 1 but is 0.5
                     function=Logistic(bias=decbias, gain=1.0),     # Set the Logistic function with bias = decbias
                     noise=NormalDist(standard_dev=SD).function,    # Set noise with seed generator compatible with MATLAB random seed generator 22 (rsg=22)
                                                                    # Please see https://github.com/jonasrauber/randn-matlab-python for further documentation
                     integrator_mode=True,
                     name='DECISION LAYER')

for output_state in decision_layer.output_states:
    output_state.value *= 0.0                                       # Set initial output values for decision layer to 0

# Create Response Layer  --- [ Target1, Target2 ]
response_layer = LCA(size=2,                                        # Number of units in input layer
                     initial_value= [[0.0,0.0]],                    # Initial input values
                     time_step_size=dt,                             # Integration step size
                     leak=-1.0,                                     # Sets off diagonals to negative values
                     self_excitation=selfrwt,                       # Set diagonals to self excitate
                     competition=respinhwt,                         # Set off diagonals to inhibit
                     function=Logistic(bias=respbias, gain=1.0),    # Set the Logistic function with bias = decbias
                     noise=NormalDist(standard_dev=SD).function,    # Set noise with seed generator compatible with MATLAB random seed generator 22 (rsg=22)
                                                                    # Please see https://github.com/jonasrauber/randn-matlab-python for further documentation
                     integrator_mode=True,
                     name='DECISION LAYER')

for output_state in response_layer.output_states:
    output_state.value *= 0.0                                       # Set initial output values for response layer to 0

# Connect mechanisms --------------------------------------------------------------------------------------------------

# Now, we create 2 weight matrices that connect the 3 behavioral layers.

# Weight matrix from Input Layer --> Decision Layer
input_weights = np.array([[inpwt, crswt, crswt],                    # Input weights are diagonals, cross weights are off diagonals
                          [crswt, inpwt, crswt],
                          [crswt, crswt, inpwt]])

# Weight matrix from Decision Layer --> Response Layer
output_weights = np.array([[decwt, 0.0],                            # Projection weight from decision layer from T1 and T2 but not distraction unit (row 3 set to all zeros) to response layer
                           [0.0, decwt],                            # Need a 3 by 2 matrix, to project from decision layer with 3 units to response layer with 2 units
                           [0.0, 0.0]])

# The process will connect the layers and weights.
decision_process = Process(pathway=[input_layer,                    # Connect the layers and weight in the order you they should be connected
                                    input_weights,
                                    decision_layer,
                                    output_weights,
                                    response_layer],
                           name='DECISION PROCESS')

# Abstracted LC to modulate gain --------------------------------------------------------------------

# This LCControlMechanism modulates gain.
LC = LCControlMechanism(integration_method="EULER",                 # We set the integration method to Euler like in the paper
                        threshold_FHN=a,                            # Here we use the Euler method for integration and we want to set the parameters,
                        uncorrelated_activity_FHN=d,                # for the FitzHugh–Nagumo system.
                        time_step_size_FHN=dt,
                        mode_FHN=C,
                        time_constant_v_FHN=tau_v,
                        time_constant_w_FHN=tau_u,
                        a_v_FHN=-1.0,
                        b_v_FHN=1.0,
                        c_v_FHN=1.0,
                        d_v_FHN=0.0,
                        e_v_FHN=-1.0,
                        f_v_FHN=1.0,
                        a_w_FHN=1.0,
                        b_w_FHN=-1.0,
                        c_w_FHN=0.0,
                        t_0_FHN=0.0,
                        base_level_gain=G,                          # Additionally, we set the parameters k and G to compute the gain equation.
                        scaling_factor_gain=k,
                        initial_v_FHN=initial_v,                    # Initialize v
                        initial_w_FHN=initial_w,                    # Initialize w (WATCH OUT !!!: In the Gilzenrat paper the authors set this parameter to be u, so that one does not think about a small w as if it would represent a weight
                        objective_mechanism=ObjectiveMechanism(function=Linear, # Project the output of T1 and T2 but not the distraction unit of the decision layer to the LC with a linear function.
                                                               monitored_output_states=[(decision_layer,
                                                                                        None,
                                                                                        None,
                                                                                        np.array([[lcwt],[lcwt],
                                                                                                  [0.0]]))],
                                                               name='LC ObjectiveMechanism'),
                        modulated_mechanisms=[decision_layer, response_layer],  # Modulate gain of decision & response layers
                        name='LC')

for output_state in LC.output_states:
	output_state.value *= G + k*initial_w          # When we run the System the very first time we need to set initial gain to G + k*initial_w, since the decison layer executes before the LC and hence needs one initial gain value to start with.


# Now, we specify the processes of the System, which in this case is just the decision_process
task = System(processes=[decision_process])

# Create Stimulus -----------------------------------------------------------------------------------------------------

# In the paper, each period has 100 time steps, so we will create 11 time periods.
# As described in the paper in figure 3, during the first 3 time periods the distractor units are given an input fixed to 1.
# Then T1 gets turned on during time period 4 with an input of 1.
# T2 gets turns on with some lag from T1 onset on, in this example we turn T2 on with Lag 2 and an input of 1
# Between T1 and T2 and after T2 the distractor unit is on.

# We create one array with 3 numbers, one for each input unit and repeat this array 100 times for one time period
# We do this 11 times. T1 is on for time4, T2 is on for time7 to model Lag3
stepSize = 100  # Each stimulus is presented for two units of time which is equivalent to 100 time steps
time1 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time2 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time3 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time4 = np.repeat(np.array([[1,0,0]]), stepSize,axis =0)    # Turn T1 on
time5 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time6 = np.repeat(np.array([[0,1,0]]), stepSize,axis =0)    # Turn T2 on --> example for Lag 2
time7 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time8 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time9 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time10 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time11 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)

# Concatenate the 11 arrays to one array with 1100 rows and 3 colons.
time = np.concatenate((time1, time2, time3, time4, time5, time6, time7, time8, time9, time10, time11), axis = 0)

# assign inputs to input_layer (Origin Mechanism) for each trial
stim_list_dict = {input_layer:time}

## Record results & run model ------------------------------------------------------------------------------------------

# Function to compute h(v) from LC's v value
def h_v(v,C,d):
    return C*v + (1-C)*d

# Initialize output arrays for plotting
LC_results_v = [h_v(initial_v,C,d)]
LC_results_w = [initial_w]
decision_layer_target = [0.5]
decision_layer_target2 = [0.5]
decision_layer_distractor = [0.5]
response = [0.5]
response2 = [0.5]
LC_gain = [0.5]
decision_layer_gain = [0.5]
decision_layer_output = [0.0]

def record_trial():
    LC_results_v.append(h_v(LC.value[2][0], C, d))
    LC_results_w.append(LC.value[3][0])
    decision_layer_target.append(decision_layer.value[0][0])
    decision_layer_target2.append(decision_layer.value[0][1])
    decision_layer_distractor.append(decision_layer.value[0][2])
    response.append(response_layer.value[0][0])
    response2.append(response_layer.value[0][1])
    LC_gain.append(LC.value[0][0])
    decision_layer_gain.append(decision_layer.function_object.gain)
    decision_layer_output.append(decision_layer.output_states[0].value)

    current_trial_num = len(LC_results_v)
    if current_trial_num%50 == 0:
        percent = int(round((float(current_trial_num) / trials)*100))
        sys.stdout.write("\r"+ str(percent) +"% complete")
        sys.stdout.flush()

sys.stdout.write("\r0% complete")
sys.stdout.flush()


# Make python seed the same as MATLAB seed
from scipy.special import erfinv
np.random.seed(22)
samples = np.random.rand(trials)
# transform from uniform to standard normal distribution using inverse cdf
samples = np.sqrt(2) * erfinv(2 * samples - 1)


## Run model ------------------------------------------------------------------------------------------

task.run(stim_list_dict, num_trials= 50, call_after_trial=record_trial)

from matplotlib import pyplot as plt
import numpy as np
t = np.arange(0.0, 1101, 1)
ax = plt.gca()
ax2 = ax.twinx()

ax.plot(t, LC_results_v, label="h(v)")
ax2.plot(t, LC_results_w, label="w", color = 'red')
# plt.plot(t, decision_layer_target, label="target")
# plt.plot(t, decision_layer_target2, label="target2")
#
# plt.plot(t,decision_layer_distractor, label="distractor")
# plt.plot(t, response_layer, label="response")
# plt.plot(t, response_layer2, label="response2")
ax.set_xlabel('Activation')
# ax.set_ylabel('h(V)')
ax.legend(loc='upper left')
ax2.legend(loc='upper left')

plt.title('Nieuwenhaus 2005 PsyNeuLink Lag 3', fontweight='bold')
ax.set_ylim((-0.2,1.0))
ax2.set_ylim((0.0, 0.4))
plt.show()


# This displays a diagram of the System
# task.show_graph()

