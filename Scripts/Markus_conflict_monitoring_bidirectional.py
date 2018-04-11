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

task_input_layer = pnl.TransferMechanism(size=2,
                                          function=pnl.Linear,
                                          name='TASK_INPUT')

#   Task layer, tasks: ('name the color', 'read the word')
#change from Linear to Logistic with control on. Linear for saniti checks
task_layer = pnl.RecurrentTransferMechanism(default_variable=np.array([[0.3, 0.05]]),
                                            size=2,
                                            auto=1,
                                            smoothing_factor=0.01,
                                   # function=pnl.Logistic(gain=(1.0, pnl.ControlProjection())),#receiver= response_layer.output_states[1],
#'DECISION_ENERGY'))
                                       #modulation=pnl.ModulationParam.OVERRIDE,#what to implement here
                                   name='TASK')


#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
colors_hidden_layer = pnl.RecurrentTransferMechanism(size=2,
                                            function=pnl.Logistic(gain=1.0, bias=4.0), # bias 4.0 is -4.0 in the paper see Docs for description
                                            integrator_mode=True,
                                                     auto=-2,
                                          #  noise=pnl.NormalDist(mean=0.0, standard_dev=.005).function,
                                            smoothing_factor=0.01, # cohen-huston text says 0.01
                                                     # "The activation function for each unit is also the same as the
                                                     # one used in the original model(the logistic of the
                                                     # time-averaged net input, with a time constant of 0.01)
                                            name='COLORS HIDDEN')

words_hidden_layer = pnl.RecurrentTransferMechanism(size=2,
                                           function=pnl.Logistic(gain=1.0, bias=4.0),
                                                    auto=-2,
                                           integrator_mode=True,
                                       #    noise=pnl.NormalDist(mean=0.0, standard_dev=.005).function,
                                           smoothing_factor=0.01,
                                           name='WORDS HIDDEN')

#Log:
#task_layer.set_log_conditions('gain')
task_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')



#   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
response_layer = pnl.RecurrentTransferMechanism(size=2,  #Recurrent
                         function=pnl.Logistic,#pnl.Stability(matrix=np.matrix([[0.0, -1.0], [-1.0, 0.0]])),
                         auto=1,
                         name='RESPONSE',
                         output_states = [pnl.RECURRENT_OUTPUT.RESULT,
                                          {pnl.NAME: 'DECISION_ENERGY',
                                           pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                           pnl.FUNCTION: pnl.Stability(default_variable=np.array([0.0, -1.0]),
                                                                       metric=pnl.ENERGY,
                                                                       # matrix=np.array([[0.0, -1.0], [-1.0, 0.0]]))}],
                                                                       matrix=np.array([[0.0, -1.0], [-1.0, 0.0]]))}],
                         integrator_mode=True,#)
                        # noise=pnl.NormalDist(mean=0.0, standard_dev=.01).function)
                         smoothing_factor=0.01)

response_layer.set_log_conditions('value')
response_layer.set_log_conditions('DECISION_ENERGY')
#response_layer.set_log_conditions('gain')


# color_word_input_weights = np.array([[1.0, 1.0],
#                                      [0.0, 0.0],
#                                      [0.0, 0.0],
#                                      [1.0, 1.0]])

color_task_weights  = np.array([[3.0, 0.0],
                                   [0.0, 3.0]])

# task_color_weights  = np.array([[3.0, 3.0],
#                                 [0, 0]])

color_response_weights = np.array([[1.5, 0.0],
                                   [0.0, 1.5]])

# response_color_weights = np.array([[1.5, 0.0],
#                                    [0.0, 1.5]])


word_task_weights   = np.array([[3.0, 0.0],
                                   [0.0, 3.0]])

# task_word_weights   = np.array([[0.0, 0.0],
#                                 [3.0, 3.0]])

word_response_weights  = np.array([[2.5, 0.0],
                                   [0.0, 2.5]])

# response_word_weights  = np.array([[2.5, 0.0],
#                                    [0.0, 2.5]])
#   CREATE PATHWAYS
#   Words pathway

#   Colors pathway


color_response_process = pnl.Process(pathway=[colors_input_layer,
                                              colors_hidden_layer,
                                              color_response_weights,
                                              response_layer],
                                     name='COLORS_RESPONSE_PROCESS')

color_control_process = pnl.Process(pathway=[colors_hidden_layer,
                                             color_task_weights,
                                             task_layer],
                                    name='COLORS_TASK_PROCESS')

word_response_process = pnl.Process(pathway=[words_input_layer,
                                             words_hidden_layer,
                                             word_response_weights,
                                             response_layer],
                                     name='COLORS_RESPONSE_PROCESS')

word_control_process = pnl.Process(pathway=[words_hidden_layer,
                                            word_task_weights,
                                            task_layer],
                                   name='WORDS_TASK_PROCESS')


#
task_color_response_process = pnl.Process(pathway=[task_input_layer,
                                                   task_layer,
                                                   color_task_weights,
                                                   colors_hidden_layer])

task_word_response_process = pnl.Process(pathway=[task_input_layer,
                                                  task_layer,
                                                   word_task_weights,
                                                   words_hidden_layer])

response_color_task_process = pnl.Process(pathway=[response_layer,
                                                   color_response_weights,
                                                   colors_hidden_layer])

response_word_task_process = pnl.Process(pathway=[response_layer,
                                                  word_response_weights,
                                                  words_hidden_layer])

#   CREATE SYSTEM
System_Conflict_Monitoring = pnl.System(processes=[color_response_process,
                                                   color_control_process,
                                                   word_response_process,
                                                   word_control_process,
                                                   task_color_response_process,
                                                   task_word_response_process,
                                                   response_color_task_process,
                                                   response_word_task_process],
                      controller=pnl.ControlMechanism,
                       monitor_for_control=[response_layer],
                       enable_controller=True,
                       name='FEEDFORWARD_STROOP_SYSTEM')

# System_Conflict_Monitoring.show_graph(show_control=pnl.ALL, show_dimensions=pnl.ALL)

# LOGGING:
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')

#   CREATE THRESHOLD FUNCTION
#first value of DDM's value is DECISION_VARIABLE
def pass_threshold(response, thresh):
    results1 = response_layer.output_states.values[0][0] #red response
    results2 = response_layer.output_states.values[0][1] #green response
    print(results1)
    print(results2)
    if results1  >= thresh or results2 >= thresh:
        return True
    # for val in results1:
    #     if val >= thresh:
    #         return True
    return False
accumulator_threshold = 0.6

terminate_trial = {
   pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, accumulator_threshold)
}

def trial_dict(red_color, green_color, red_word, green_word, CN, WR):

    trialdict = {
    colors_input_layer: [red_color, green_color],
    words_input_layer: [red_word, green_word],
    task_input_layer: [CN, WR]
    }
    return trialdict

# Define initialization trials separately
# input just task and run once so system asymptotes
WR_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 1)

CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0)

#
System_Conflict_Monitoring.run(inputs=CN_trial_initialize_input, num_trials=3)#, termination_processing=terminate_trial)





# #function to test a particular trial type
# def testtrialtype(test_trial_input, initialize_trial_input, ntrials):#, plot_title, trial_test_counter):
#     # create variable to store results
#     results = np.empty((1, 0))
#     decision_energy = np.empty((ntrials))
#     response_activity1 = np.empty((ntrials))
#     response_activity2 = np.empty((ntrials))
#
#     # clear log
#     # respond_red_accumulator.log.clear_entries(delete_entry=False)
#     # respond_red_accumulator.reinitialize(0)
#     # respond_green_accumulator.reinitialize(0)
#     response_layer.reinitialize([[0, 0]])
#
#     for trial in range(ntrials):
#         # run system once (with integrator mode off and no noise for hidden units) with only task so asymptotes
#         colors_hidden_layer.integrator_mode = False
#         words_hidden_layer.integrator_mode = False
#         response_layer.integrator_mode = False
#         # colors_hidden_layer.noise = 0
#         # words_hidden_layer.noise = 0
#         # response_layer.noise = 0
#
#         System_Conflict_Monitoring.run(inputs=initialize_trial_input, termination_processing=terminate_trial)
#         # but didn't want to run accumulators so set those back to zero
#         # respond_green_accumulator.reinitialize(0)
#         # respond_red_accumulator.reinitialize(0)
#         # print('response layer value: ', response_layer.output_states[1].value)
#         # now put back in integrator mode and noise
#         colors_hidden_layer.integrator_mode = True
#         words_hidden_layer.integrator_mode = True
#         response_layer.integrator_mode = True
#         #colors_hidden_layer.noise = pnl.NormalDist(mean=0, standard_dev=unit_noise).function
#         #words_hidden_layer.noise = pnl.NormalDist(mean=0, standard_dev=unit_noise).function
#         #response_layer.noise = pnl.NormalDist(mean=0, standard_dev=unit_noise).function
#
#         # run system with test pattern
#         #System_Conflict_Monitoring.run(inputs=test_trial_input)
#
#         # store results
#         # my_results = response_layer.log.nparray_dictionary()
#         # print('respond_red_accumulator.log.nparray_dictionary(): ',respond_red_accumulator.log.nparray_dictionary())
#         # how many cycles to run? count the length of the log
#         # num_timesteps = np.asarray(np.size((my_results['DECISION_ENERGY'])/2))
#         # print('num_timesteps; ', num_timesteps)
#
#         # value of parts of the system
#         # decision_energy[trial] = np.asarray(response_layer.output_states[1].value)
#         # # print('decision_energy; ', decision_energy[trial])
#         # response_activity1[trial] = np.asarray(response_layer.output_states[0].value[0])
#         # response_activity2[trial] = np.asarray(response_layer.output_states[0].value[1])
#
#
#
#         # print('num_timesteps: ', num_timesteps)
#         # print('respond_red: ', respond_red)
#         # print('red_activity: ', red_activity)
#         # print('green_activity: ', green_activity)
#         # print('colors_hidden_layer_value: ', colors_hidden_layer_value)
#         # print('words_hidden_layer_value: ', words_hidden_layer_value)
#         # print('response_layer_value: ', response_layer_value)
#         tmp_results = np.concatenate((decision_energy,
#                                       response_activity1,
#                                       response_activity2), axis=0)
#
#
#         # print('tmp_results: ', tmp_results)
#
#         # after a run we want to reset the activations of the integrating units so we can test many trials and examine the distrubtion of responses
#         #words_hidden_layer.reinitialize([0, 0])
#         #colors_hidden_layer.reinitialize([0, 0])
#         #response_layer.reinitialize([0, 0])
#         # clear log to get num_timesteps for next run
#         #response_layer.log.clear_entries(delete_entry=False)
#         results = tmp_results #decision_energy
#
#         # results = np.append(results, tmp_results, axis=1)
#
#         # print('tmp_results: ', tmp_results)
#
#         # after a run we want to reset the activations of the integrating units so we can test many trials and examine the distrubtion of responses
#         # words_hidden_layer.reinitialize([0, 0])
#         # colors_hidden_layer.reinitialize([0, 0])
#         # response_layer.reinitialize([0, 0])
#         # clear log to get num_timesteps for next run
#         # respond_red_accumulator.log.clear_entries(delete_entry=False)
#
#
#
#     return results
#
#
#
# # trial_test_counter = 1
# #test WR control trial
# # ntrials = 50
# # WR_control_trial_title = 'RED word (control) WR trial where Red correct'
# # WR_control_trial_input = trial_dict(0, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# # results_WR_control_trial = testtrialtype(WR_control_trial_input,
# #                                          WR_trial_initialize_input,
# #                                          ntrials,
# #                                          WR_control_trial_title,
# #                                          trial_test_counter)
#
# # ntrials = 50
# # WR_congruent_trial_title = 'congruent WR trial where Red correct'
# # WR_congruent_trial_input = trial_dict(1, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
# # results_WR_congruent_trial = testtrialtype(WR_congruent_trial_input,
# #                                            WR_trial_initialize_input,
# #                                            ntrials,
# #                                            WR_congruent_trial_title,
# #                                            trial_test_counter)
#
# # ntrials = 150
# # WR_incongruent_trial_title = 'incongruent WR trial where Red correct'
# # WR_incongruent_trial_input = trial_dict(1, 0, 0, 1, 0, 1) #red_color, green color, red_word, green word, CN, WR
# # results_WR_incongruent_trial = testtrialtype(WR_incongruent_trial_input,
# #                                              WR_trial_initialize_input,
# #                                              ntrials,
# #                                              WR_incongruent_trial_title,
# #                                              trial_test_counter)
# #
# # print(response_layer.value)
#
#
# ntrials = 1
# # CN_control_trial_input = trial_dict(1, 0, 0, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
# # results_CN_control_trial = testtrialtype(CN_control_trial_input,
# #                                          CN_trial_initialize_input,
# #                                          ntrials)
# # # ntrials = 50
# # CN_congruent_trial_input = trial_dict(1, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
# # results_CN_congruent_trial = testtrialtype(CN_congruent_trial_input,
# #                                            CN_trial_initialize_input,
# #                                            ntrials)
#
# # # ntrials = 50
# CN_incongruent_trial_input = trial_dict(1, 0, 0, 1, 1, 0) #red_color, green color, red_word, green word, CN, WR
# results_CN_incongruent_trial = testtrialtype(CN_incongruent_trial_input,
#                                              CN_trial_initialize_input,
#                                              ntrials)
#
#
#
# legend = ['control',
#         'congruent',
#         'incongruent']
# colors = ['b', 'c', 'r']
# t = np.arange(1.0, ntrials+1, 1.0)
# plt.plot(t,results_CN_control_trial[0:ntrials], 'b')
# plt.plot(t,results_CN_congruent_trial[0:ntrials],'r')
# plt.plot(t,results_CN_incongruent_trial[0:ntrials], 'g')
#
# plt.tick_params(axis='x', labelsize=9)
# plt.title('Conflict Monitoring')
# plt.legend(legend)
# #
#
# plt.show()
# #
# # plt.plot(t,results_CN_control_trial[ntrials:ntrials*2], 'b')
# # plt.plot(t,results_CN_control_trial[2*ntrials:ntrials*3], 'c')
# #
# # plt.plot(t,results_CN_congruent_trial[ntrials:ntrials*2], 'orange')
# # plt.plot(t,results_CN_congruent_trial[2*ntrials:ntrials*3], 'r')
# #
# # plt.plot(t,results_CN_incongruent_trial[ntrials:ntrials*2], 'green')
# # plt.plot(t,results_CN_incongruent_trial[2*ntrials:ntrials*3], 'green')
# # plt.show()
#
# # plt.plot(t,results_CN_congruent_trial[ntrials:ntrials*2],'r')
# # plt.plot(t,results_CN_incongruent_trial[ntrials:ntrials*2], 'g')
#
# print('decision_energy for control :', results_CN_control_trial)
# #
# # plt.plot(cycles_x[0:3], cycles_mean[0:3], color=colors[0])
# # plt.errorbar(cycles_x[0:3], cycles_mean[0:3], xerr=0, yerr=cycles_std[0:3], ecolor=colors[0], fmt='none')
# # plt.scatter(cycles_x[0], cycles_mean[0], marker='x', color=colors[0])
# # plt.scatter(cycles_x[1], cycles_mean[1], marker='x', color=colors[0])
# # plt.scatter(cycles_x[2], cycles_mean[2], marker='x', color=colors[0])
# # plt.plot(cycles_x[3:6], cycles_mean[3:6], color=colors[1])
# # plt.errorbar(cycles_x[3:6], cycles_mean[3:6], xerr=0, yerr=cycles_std[3:6], ecolor=colors[1], fmt='none')
# # plt.scatter(cycles_x[3], cycles_mean[3], marker='o', color=colors[1])
# # plt.scatter(cycles_x[4], cycles_mean[4], marker='o', color=colors[1])
# # plt.scatter(cycles_x[5], cycles_mean[5], marker='o', color=colors[1])
# #
#
#
#
#
# #my_Stroop.run(inputs=CN_trial_initialize_input,
#  #           )
# # response_layer.log.print_entries()
#
# # print('response value: ', response_layer.output_states.values)
#
#
# # my_Stroop.run(inputs=test_trial_input)
# # #run system once with only task so asymptotes
# # nTrials = 5
# # #my_Stroop.run(inputs=CN_incongruent_trial_input, num_trials=nTrials)
# # #but didn't want to run accumulators so set those back to zero
# # #respond_green_accumulator.reinitialize(0)
# # #respond_red_accumulator.reinitialize(0)
# # # now run test trial
# # #my_Stroop.show_graph(show_mechanism_structure=pnl.VALUES)
# # my_Stroop.run(inputs=CN_incongruent_trial_input, num_trials=1)
# #
# # response_layer.log.print_entries()
# # my_Stroop.log.print_entries()
#
