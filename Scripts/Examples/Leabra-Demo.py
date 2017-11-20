# The following script briefly demos the LeabraMechanism in PsyNeuLink by comparing its output with a corresponding
# network from the leabra package.
# Before running this, please make sure you are using Python 3.5, and that you have installed the leabra package in
# your Python 3.5 environment.

# Installation notes:
#
# If you see an error such as:
#  "Runtime warning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88"
# then this may be an issue with scipy (or other similar modules such as scikit-learn or sklearn).
#
# To resolve this, if you have pip, then use PyCharm to uninstall scipy (or other packages if they continue
# to give you trouble) and then use "pip install scipy --no-use-wheel". Or, if you can figure out how to get PyCharm
#  to ignore warnings, that's fine too.
#
# More info here: https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate
# -binary-incompatibility

import warnings
warnings.filterwarnings("ignore", message=r".*numpy.dtype size changed.*")
import numpy as np
import random
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     import leabra
from psyneulink.library.mechanisms.processing.leabramechanism import LeabraMechanism, build_network, train_network
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.functions.function import Linear
from psyneulink.components.process import Process
from psyneulink.components.system import System
import time

random_seed_value = 1  # feel free to change this value
random.seed(random_seed_value)  # setting the random seed ensures the two Leabra networks are identical (see line 68)
num_trials = 10  # how many trials should we run?
hidden_layers = 4  # how many hidden layers are there?
hidden_sizes = [2, 3, 4, 5]  # how big is each hidden layer?
input_pattern = np.repeat(np.array([[0, 1, 3, 4]]), num_trials, axis=0)  # the input
print("inputs to the networks will be: ", input_pattern)
# similar example: input_pattern = [[0, 1, 3, 4]] * int(num_trials/2) + [[0, 0, 0, 0]] * int(num_trials/2)
training_pattern = np.repeat(np.array([[0, 0, 0]]), num_trials, axis=0)  # the training pattern
print("training inputs to the networks will be: ", training_pattern)
input_size = len(input_pattern[0])  # how big is the input layer of the network?
output_size = len(training_pattern[0])  # how big is the output layer of the network?
train_flag = False  # should the LeabraMechanism and leabra network learn?
# NOTE: there is currently a bug with training, in which the output may differ between trials, randomly
# ending up in one of two possible outputs. Running this script repeatedly will make this behavior clear.
# The leabra network and LeabraMechanism experience this bug equally.

# NOTE: The reason TransferMechanisms are used below is because there is currently a bug where LeabraMechanism
# (and other `Mechanism`s with multiple input states) cannot be used as origin Mechanisms for a System. If you desire
# to use a LeabraMechanism as an origin Mechanism, you can work around this bug by creating two `TransferMechanism`s
# as origin Mechanisms instead, and have these two TransferMechanisms pass their output to the InputStates of
# the LeabraMechanism.

# create a LeabraMechanism in PsyNeuLink
L = LeabraMechanism(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers,
                     hidden_sizes=hidden_sizes, name='L', training_flag=train_flag)


T1 = TransferMechanism(name='T1', size=input_size, function=Linear)
T2 = TransferMechanism(name='T2', size=output_size, function=Linear)

p1 = Process(pathway=[T1, L])
proj = MappingProjection(sender=T2, receiver=L.input_states[1])
p2 = Process(pathway=[T2, proj, L])
s = System(processes=[p1, p2])

print('Running Leabra in PsyNeuLink...')
start_time = time.process_time()
outputs = s.run(inputs={T1: input_pattern.copy(), T2: training_pattern.copy()})
end_time = time.process_time()

print('Time to run LeabraMechanism in PsyNeuLink: ', end_time - start_time, "seconds")
print('LeabraMechanism Outputs Over Time: ', outputs, type(outputs))
print('LeabraMechanism Final Output: ', outputs[-1], type(outputs[-1]))


random.seed(random_seed_value)
leabra_net = build_network(n_input=input_size, n_output=output_size, n_hidden=hidden_layers, hidden_sizes=hidden_sizes,
                           training_flag=train_flag)

print('\nRunning Leabra in Leabra...')
start_time = time.process_time()
for i in range(num_trials):
    train_network(leabra_net, input_pattern[i].copy(), training_pattern[i].copy())
end_time = time.process_time()
print('Time to run Leabra on its own: ', end_time - start_time, "seconds")
print('Leabra Output: ', [unit.act_m for unit in leabra_net.layers[-1].units], type([unit.act_m for unit in leabra_net.layers[-1].units][0]))