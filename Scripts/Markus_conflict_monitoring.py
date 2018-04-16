import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl


LAMBDA = 0.95
alpha = 11.24
beta = 9.46

#Conflict  equation:

#C(t+1) = LAMBDA*C(t) +(1-LAMBDA) * (alpha*ENERGY(t) + beta)



# SET UP MECHANISMS
#   Linear input units, colors: ('red', 'green'), words: ('RED','GREEN')
colors_input_layer = pnl.TransferMechanism(size=2,
                                           function=pnl.Linear,
                                           name='COLORS_INPUT')

words_input_layer = pnl.TransferMechanism(size=2,
                                          function=pnl.Linear,
                                          name='WORDS_INPUT')

#   Task layer, tasks: ('name the color', 'read the word')
task_layer = pnl.TransferMechanism(size=2,
                                   # function= pnl.Linear(),
                                   function=pnl.Logistic(gain=(1.0, pnl.ControlProjection())),
                                   name='TASK')

#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
colors_hidden_layer = pnl.TransferMechanism(size=2,
                                            function=pnl.Logistic(gain=1.0, bias=4.0), # bias 4.0 is -4.0 in the paper see Docs for description
                                            integrator_mode=True,
                                                     # auto=-1,
                                          #  noise=pnl.NormalDist(mean=0.0, standard_dev=.005).function,
                                            smoothing_factor=0.1,
                                            name='COLORS HIDDEN')

words_hidden_layer = pnl.TransferMechanism(size=2,
                                           function=pnl.Logistic(gain=1.0, bias=4.0),
                                                    # auto=-1,
                                           integrator_mode=True,
                                       #    noise=pnl.NormalDist(mean=0.0, standard_dev=.005).function,
                                           smoothing_factor=0.1,
                                           name='WORDS HIDDEN')

#Log:
# task_layer.set_log_conditions('gain')
task_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')

#   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
response_layer = pnl.RecurrentTransferMechanism(size=2, #pnl.LCA(size=2,#
                         function=pnl.Logistic,
                         auto=0.0,
                         name='RESPONSE',
                         output_states = [pnl.RECURRENT_OUTPUT.RESULT,
                                          {pnl.NAME: 'DECISION_ENERGY',
                                           pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                           pnl.FUNCTION: pnl.Stability(default_variable=np.array([0.0, 0.0]),
                                                                       metric=pnl.ENERGY,
                                                                       matrix=np.array([[0.0, -1.0], [-1.0, 0.0]]))}],
                         integrator_mode=True,
                        # noise=pnl.NormalDist(mean=0.0, standard_dev=.01).function)
                         smoothing_factor=0.1)

response_layer.set_log_conditions('value')
response_layer.set_log_conditions('DECISION_ENERGY')
# response_layer.set_log_conditions('gain')

# >>> L = pnl.Logistic(gain = 2)
# >>> def my_fct(variable):
# ...     return L.function(variable) + 2
# >>> my_mech = pnl.ProcessingMechanism(size = 3, function = my_fct)
# >>> my_mech.execute(input = [1, 2, 3])
# array([[2.88079708, 2.98201379, 2.99752738]])

#C(t+1) = LAMBDA*C(t) +(1-LAMBDA) * (alpha*ENERGY(t) + beta)


I = pnl.Integrator(rate= 0.95,
                          noise = 0.0)
Linear = pnl.Linear(slope = 11.24, intercept= 9.46)

def my_fct(variable1, variable2):
    return I.function(variable1) + (1-0.95) * Linear.function(variable2)

my_mech = pnl.ProcessingMechanism(function=my_fct())

#
# conflict = pnl.IntegratorMechanism(function=pnl.AdaptiveIntegrator(rate=0.95))
# conflict.set_log_conditions('value')
#


#   SET UP CONNECTIONS
# column 0: input_'red' to hidden_'red', hidden_'green'
# column 1: input_'green' to hidden_'red', hidden_'green'
color_weights = pnl.MappingProjection(matrix=np.matrix([[2.2, -2.2],
                                                        [-2.2, 2.2]]),
                                      name='COLOR_WEIGHTS')
# column 0: input_'RED' to hidden_'RED', hidden_'GREEN'
# column 1: input_'GREEN' to hidden_'RED', hidden_'GREEN'
word_weights = pnl.MappingProjection(matrix=np.matrix([[2.6, -2.6],
                                                       [-2.6, 2.6]]),
                                     name='WORD_WEIGHTS')

#   Hidden to response
# column 0: hidden_'red' to response_'red', response_'green'
# column 1: hidden_'green' to response_'red', response_'green'
color_response_weights = pnl.MappingProjection(matrix=np.matrix([[1.3, -1.3],
                                                                 [-1.3, 1.3]]),
                                               name='COLOR_RESPONSE_WEIGHTS')

# column 0: hidden_'RED' to response_'red', response_'green'
# column 1: hidden_'GREEN' to response_'red', response_'green'
word_response_weights = pnl.MappingProjection(matrix=np.matrix([[2.5, -2.5],
                                                                [-2.5, 2.5]]),
                                              name='WORD_RESPONSE_WEIGHTS')

#   Task to hidden layer
# column 0: task_CN to hidden_'red', hidden_'green'
# column 1: task_WR to hidden_'red', hidden_'green'
task_CN_weights = pnl.MappingProjection(matrix=np.matrix([[4.0, 4.0],
                                                          [0.0, 0.0]]),
                                        name='TASK_CN_WEIGHTS')

# column 0: task_CN to hidden_'RED', hidden_'GREEN'
# column 1: task_WR to hidden_'RED', hidden_'GREEN'
task_WR_weights = pnl.MappingProjection(matrix=np.matrix([[0, 0.0],
                                                          [4.0, 4.0]]),
                                        name='TASK_WR_WEIGHTS')

#   CREATE PATHWAYS
#   Words pathway
words_process = pnl.Process(pathway=[words_input_layer,
                                     word_weights,
                                     words_hidden_layer,
                                     word_response_weights,
                                     response_layer], name='WORDS_PROCESS')

#   Colors pathway
colors_process = pnl.Process(pathway=[colors_input_layer,
                                      color_weights,
                                      colors_hidden_layer,
                                      color_response_weights,
                                      response_layer], name='COLORS_PROCESS')

#   Task representation pathway
task_CN_process = pnl.Process(pathway=[task_layer,
                                       task_CN_weights,
                                       colors_hidden_layer],
                              name='TASK_CN_PROCESS')

task_WR_process = pnl.Process(pathway=[task_layer,
                                       task_WR_weights,
                                       words_hidden_layer],
                              name='TASK_WR_PROCESS')

#   CREATE SYSTEM
System_Conflict_Monitoring = pnl.System(processes=[colors_process,
                                  words_process,
                                  task_CN_process,
                                  task_WR_process],
                      controller=pnl.ControlMechanism,
                       monitor_for_control=[response_layer.output_states['DECISION_ENERGY']],
                       enable_controller=True,
                       name='FEEDFORWARD_Conflict_Monitoring_SYSTEM')

# System_Conflict_Monitoring.show_graph(show_control=pnl.ALL, show_dimensions=pnl.ALL)


# System_Conflict_Monitoring.controller.set_log_conditions('TASK[gain] ControlSignal', pnl.LogCondition.EXECUTION)
# System_Conflict_Monitoring.controller.set_log_conditions('value')
# System_Conflict_Monitoring.controller.set_log_conditions('slope')
# System_Conflict_Monitoring.controller.set_log_conditions('intercept')


def trial_dict(red_color, green_color, red_word, green_word, CN, WR):

    trialdict = {
    colors_input_layer: [red_color, green_color],
    words_input_layer: [red_word, green_word],
    task_layer: [CN, WR]
    }
    return trialdict

# Define initialization trials separately
# input just task and run once so system asymptotes
WR_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 1)

CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 1, 0)

# function to test a particular trial type
def testtrialtype(test_trial_input, initialize_trial_input, ntrials):  # , plot_title, trial_test_counter):
    # create variable to store results
    decision_energy = np.empty((ntrials))
    response_activity1 = np.empty((ntrials))
    response_activity2 = np.empty((ntrials))
    task_layer1 = np.empty((ntrials))
    task_layer2 = np.empty((ntrials))
    colors_hidden_layer1 = np.empty((ntrials))
    colors_hidden_layer2 = np.empty((ntrials))
    response_layer.reinitialize([[0, 0]])
    words_hidden_layer.reinitialize([[0, 0]])
    colors_hidden_layer.reinitialize([[0, 0]])
    words_hidden_layer1 = np.empty((ntrials))
    words_hidden_layer2 = np.empty((ntrials))

    # run system once with integrator mode off and only task so asymptotes
    colors_hidden_layer.integrator_mode = False
    words_hidden_layer.integrator_mode = False
    response_layer.integrator_mode = False

    #     System_Conflict_Monitoring.run(inputs=initialize_trial_input)

    # Turn integrator mode on
    colors_hidden_layer.integrator_mode = True
    words_hidden_layer.integrator_mode = True
    response_layer.integrator_mode = True

    for trial in range(ntrials):
        # run system with test pattern
        System_Conflict_Monitoring.run(inputs=test_trial_input)

        # value of parts of the system
        decision_energy[trial] = np.asarray(response_layer.output_states[1].value)
        response_activity1[trial] = np.asarray(response_layer.output_states[0].value[0])
        response_activity2[trial] = np.asarray(response_layer.output_states[0].value[1])

        task_layer1[trial] = np.asarray(task_layer.output_states[0].value[0])
        task_layer2[trial] = np.asarray(task_layer.output_states[0].value[1])

        colors_hidden_layer1[trial] = np.asarray(colors_hidden_layer.output_states[0].value[0])
        colors_hidden_layer2[trial] = np.asarray(colors_hidden_layer.output_states[0].value[1])
        words_hidden_layer1[trial] = np.asarray(words_hidden_layer.output_states[0].value[0])
        words_hidden_layer2[trial] = np.asarray(words_hidden_layer.output_states[0].value[1])

    results = np.concatenate((decision_energy,
                              response_activity1,
                              response_activity2,
                              task_layer1,
                              task_layer2,
                              colors_hidden_layer1,
                              colors_hidden_layer2,
                              words_hidden_layer1,
                              words_hidden_layer2), axis=0)

    return results


# Run Models
ntrials = 50
CN_control_trial_input = trial_dict(1, 0, 0, 0, 1, 0) #red ink, green ink, red_word, green word, CN, WR
results_CN_control_trial = testtrialtype(CN_control_trial_input,
                                         CN_trial_initialize_input,
                                         ntrials)

CN_congruent_trial_input = trial_dict(1, 0, 1, 0, 1, 0) #red ink, green ink, red_word, green word, CN, WR
results_CN_congruent_trial = testtrialtype(CN_congruent_trial_input,
                                           CN_trial_initialize_input,
                                           ntrials)

CN_incongruent_trial_input = trial_dict(1, 0, 0, 1, 1, 0) #red ink, green ink, red_word, green word, CN, WR
results_CN_incongruent_trial = testtrialtype(CN_incongruent_trial_input,
                                             CN_trial_initialize_input,
                                             ntrials)

# Create Plots of results

#
plt.figure()
legend = ['control',
        'congruent',
        'incongruent']
colors = ['k', 'm', 'b']
t = np.arange(1.0, ntrials+1, 1.0)
plt.plot(t,results_CN_control_trial[0:ntrials], 'k')
plt.plot(t,results_CN_congruent_trial[0:ntrials],'m')
plt.plot(t,results_CN_incongruent_trial[0:ntrials], 'b')

plt.tick_params(axis='x', labelsize=9)
plt.title('Decision Energy (Measure of Response Conflict)')
plt.legend(legend)
plt.xlabel('time steps')
plt.ylabel('ENERGY')
plt.show()

plt.figure()
# colors = ['b', 'b', 'c', 'c', 'r', 'r']
legend = ['control red color unit',
          'congruent (red word) red color unit',
          'incongruent (green word) red color unit',
          'incongruent (green word) green color unit',
          'control red color unit',
          'congruent (red word) green color unit']
colors = ['b', 'g', 'r', 'lime', 'aqua', 'salmon']

plt.plot(t,results_CN_control_trial[ntrials:ntrials*2], 'k')
plt.plot(t,results_CN_congruent_trial[ntrials:ntrials*2], 'm')
plt.plot(t,results_CN_incongruent_trial[ntrials:ntrials*2], 'blue')

plt.plot(t,results_CN_incongruent_trial[2*ntrials:ntrials*3], 'aqua')
plt.plot(t,results_CN_control_trial[2*ntrials:ntrials*3], 'dimgray')
plt.plot(t,results_CN_congruent_trial[2*ntrials:ntrials*3], 'violet')
plt.title('Hidden layer activity')
plt.tick_params(axis='x', labelsize=9)
plt.title('Response Layer Units Activity (red ink input, task = ink color naming)')
plt.ylabel('Activity')
plt.xlabel('cycles')
plt.ylim(0.2,0.9)

plt.legend(legend)
plt.show()


# conflict.log.print_entries()