import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl

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
                                   function= pnl.Linear(),
                                   # function=pnl.Logistic(gain=(1.0, pnl.ControlProjection())),
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
                      # controller=pnl.ControlMechanism,
                      #  monitor_for_control=[response_layer.output_states['DECISION_ENERGY']],
                      #  enable_controller=True,
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


#function to test a particular trial type
def testtrialtype(test_trial_input, initialize_trial_input, ntrials):#, plot_title, trial_test_counter):
    # create variable to store results
    results = np.empty((1, 0))
    decision_energy = np.empty((ntrials))
    response_activity1 = np.empty((ntrials))
    response_activity2 = np.empty((ntrials))
    task_layer1 = np.empty((ntrials))
    task_layer2 = np.empty((ntrials))
    colors_hidden_layer1 = np.empty((ntrials))
    colors_hidden_layer2 = np.empty((ntrials))
    response_layer.reinitialize([[0, 0]])
    words_hidden_layer1 = np.empty((ntrials))
    words_hidden_layer2 = np.empty((ntrials))


    System_Conflict_Monitoring.run(inputs=initialize_trial_input)
    colors_hidden_layer.integrator_mode = False
    words_hidden_layer.integrator_mode = False
    response_layer.integrator_mode = False

    for trial in range(ntrials):
        # run system once (with integrator mode off and no noise for hidden units) with only task so asymptotes

        # colors_hidden_layer.noise = 0
        # words_hidden_layer.noise = 0
        # response_layer.noise = 0

        # print('response layer value: ', response_layer.output_states[1].value)
        # now put back in integrator mode and noise
        colors_hidden_layer.integrator_mode = True
        words_hidden_layer.integrator_mode = True
        response_layer.integrator_mode = True
        #colors_hidden_layer.noise = pnl.NormalDist(mean=0, standard_dev=unit_noise).function
        #words_hidden_layer.noise = pnl.NormalDist(mean=0, standard_dev=unit_noise).function
        #response_layer.noise = pnl.NormalDist(mean=0, standard_dev=unit_noise).function

        # run system with test pattern
        System_Conflict_Monitoring.run(inputs=test_trial_input)

        # value of parts of the system
        decision_energy[trial] = np.asarray(response_layer.output_states[1].value)
        # print('decision_energy; ', decision_energy[trial])
        response_activity1[trial] = np.asarray(response_layer.output_states[0].value[0])
        response_activity2[trial] = np.asarray(response_layer.output_states[0].value[1])

        task_layer1[trial] = np.asarray(task_layer.output_states[0].value[0])
        task_layer2[trial] = np.asarray(task_layer.output_states[0].value[1])

        colors_hidden_layer1[trial] = np.asarray(colors_hidden_layer.output_states[0].value[0])
        colors_hidden_layer2[trial] = np.asarray(colors_hidden_layer.output_states[0].value[1])
        words_hidden_layer1[trial] = np.asarray(words_hidden_layer.output_states[0].value[0])
        words_hidden_layer2[trial] = np.asarray(words_hidden_layer.output_states[0].value[1])

        # print('colors_hidden_layer_value: ', colors_hidden_layer_value)
        # print('words_hidden_layer_value: ', words_hidden_layer_value)
        # print('response_layer_value: ', response_layer_value)
        tmp_results = np.concatenate((decision_energy,
                                      response_activity1,
                                      response_activity2,
                                      task_layer1,
                                      task_layer2,
                                      colors_hidden_layer1,
                                      colors_hidden_layer2,
                                      words_hidden_layer1,
                                      words_hidden_layer2), axis=0)

        results = tmp_results #decision_energy

    return results



# trial_test_counter = 1
#test WR control trial
# ntrials = 50
# WR_control_trial_title = 'RED word (control) WR trial where Red correct'
# WR_control_trial_input = trial_dict(0, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# results_WR_control_trial = testtrialtype(WR_control_trial_input,
#                                          WR_trial_initialize_input,
#                                          ntrials,
#                                          WR_control_trial_title,
#                                          trial_test_counter)

# ntrials = 50
# WR_congruent_trial_title = 'congruent WR trial where Red correct'
# WR_congruent_trial_input = trial_dict(1, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# results_WR_congruent_trial = testtrialtype(WR_congruent_trial_input,
#                                            WR_trial_initialize_input,
#                                            ntrials,
#                                            WR_congruent_trial_title,
#                                            trial_test_counter)

# ntrials = 150
# WR_incongruent_trial_title = 'incongruent WR trial where Red correct'
# WR_incongruent_trial_input = trial_dict(1, 0, 0, 1, 0, 1) #red_color, green color, red_word, green word, CN, WR
# results_WR_incongruent_trial = testtrialtype(WR_incongruent_trial_input,
#                                              WR_trial_initialize_input,
#                                              ntrials,
#                                              WR_incongruent_trial_title,
#                                              trial_test_counter)
#
# print(response_layer.value)


ntrials = 50
CN_control_trial_input = trial_dict(1, 0, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
results_CN_control_trial = testtrialtype(CN_control_trial_input,
                                         CN_trial_initialize_input,
                                         ntrials)
# ntrials = 50
CN_congruent_trial_input = trial_dict(1, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
results_CN_congruent_trial = testtrialtype(CN_congruent_trial_input,
                                           CN_trial_initialize_input,
                                           ntrials)

# # ntrials = 50
CN_incongruent_trial_input = trial_dict(1, 0, 0, 1, 1, 0) #red_color, green color, red_word, green word, CN, WR
results_CN_incongruent_trial = testtrialtype(CN_incongruent_trial_input,
                                             CN_trial_initialize_input,
                                             ntrials)
# System_Conflict_Monitoring.show_graph(show_mechanism_structure=pnl.VALUES, show_control=True)


plt.figure()
legend = ['control',
        'incongruent',
        'congruent']
colors = ['b', 'g', 'r']
t = np.arange(1.0, ntrials+1, 1.0)
plt.plot(t,results_CN_control_trial[0:ntrials], 'b')
plt.plot(t,results_CN_congruent_trial[0:ntrials],'r')
plt.plot(t,results_CN_incongruent_trial[0:ntrials], 'g')

plt.tick_params(axis='x', labelsize=9)
plt.title('Conflict Monitoring')
plt.legend(legend)
plt.xlabel('trials')
plt.ylabel('ENERGY')
plt.show()

plt.figure()
# colors = ['b', 'b', 'c', 'c', 'r', 'r']
legend = ['control green color',
          'congruent green color',
          'incongruent green color',
          'incongruent red color',
          'congruent red color',
          'control red color']
colors = ['b', 'g', 'r', 'lime', 'aqua', 'salmon']

plt.plot(t,results_CN_control_trial[ntrials:ntrials*2], 'b')
plt.plot(t,results_CN_congruent_trial[ntrials:ntrials*2], 'r')
plt.plot(t,results_CN_incongruent_trial[ntrials:ntrials*2], 'g')

plt.plot(t,results_CN_incongruent_trial[2*ntrials:ntrials*3], 'lime')
plt.plot(t,results_CN_control_trial[2*ntrials:ntrials*3], 'aqua')
plt.plot(t,results_CN_congruent_trial[2*ntrials:ntrials*3], 'salmon')
plt.title('Hidden layer activity')
plt.tick_params(axis='x', labelsize=9)
plt.title('Response Layer - green color input')
plt.ylabel('Activity')
plt.xlabel('cycles')

plt.legend(legend)
plt.show()

plt.figure()
plt.plot(results_CN_control_trial[ntrials:ntrials*2], 'r')
plt.plot(results_CN_control_trial[ntrials*2:ntrials*3], 'r')
plt.title('log for response layer decision energy first 50 trial')

plt.figure()

plt.plot(results_CN_control_trial[ntrials*3:ntrials*4], 'g')
# plt.plot(results_CN_control_trial[ntrials*4:ntrials*5], 'g')

# plt.plot(task_layer_word, 'r')

plt.title('Input to green unit of task layer')

plt.figure()

plt.plot(results_CN_incongruent_trial[ntrials*7:ntrials*8], 'r')
plt.plot(results_CN_incongruent_trial[ntrials*8:ntrials*9], 'g')
plt.title('word hidden unit activity')
legend = ['red word',
          'green word']
colors = ['g', 'r']
plt.legend(legend)
