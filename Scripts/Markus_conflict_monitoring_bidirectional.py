import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl

# SET UP MECHANISMS
#   Linear input units, colors: ('red', 'green'), words: ('RED','GREEN')
colors_input_layer = pnl.TransferMechanism(size=3,
                                           function=pnl.Linear,
                                           name='COLORS_INPUT')

words_input_layer = pnl.TransferMechanism(size=3,
                                          function=pnl.Linear,
                                          name='WORDS_INPUT')

task_input_layer = pnl.TransferMechanism(size=2,
                                          function=pnl.Linear,
                                          name='TASK_INPUT')

#   Task layer, tasks: ('name the color', 'read the word')
#change from Linear to Logistic with control on. Linear for saniti checks
task_layer = pnl.RecurrentTransferMechanism(default_variable=np.array([[0, 0]]),
                                            function=pnl.Logistic(),
                                            size=2,
                                            auto=-2,
                                            smoothing_factor=0.1,
                                   # function=pnl.Logistic(gain=(1.0, pnl.ControlProjection())),#receiver= response_layer.output_states[1],
#'DECISION_ENERGY'))
                                       #modulation=pnl.ModulationParam.OVERRIDE,#what to implement here
                                   name='TASK')


#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
colors_hidden_layer = pnl.RecurrentTransferMechanism(#default_variable=np.array([[0, 0, 0]]),
                                                     size=3,
                                            function=pnl.Logistic(gain=1.0, bias=4.0), # bias 4.0 is -4.0 in the paper see Docs for description
                                            integrator_mode=True,
                                                     auto=-2,
                                          #  noise=pnl.NormalDist(mean=0.0, standard_dev=.005).function,
                                            smoothing_factor=0.1, # cohen-huston text says 0.01
                                                     # "The activation function for each unit is also the same as the
                                                     # one used in the original model(the logistic of the
                                                     # time-averaged net input, with a time constant of 0.01)
                                            name='COLORS HIDDEN')

words_hidden_layer = pnl.RecurrentTransferMechanism(#default_variable=np.array([[1, 1]]),
                                                    size=3,
                                           function=pnl.Logistic(gain=1.0, bias=4.0),
                                                    auto=-2,
                                           integrator_mode=True,
                                       #    noise=pnl.NormalDist(mean=0.0, standard_dev=.005).function,
                                           smoothing_factor=0.1,
                                           name='WORDS HIDDEN')

#Log:
#task_layer.set_log_conditions('gain')
task_layer.set_log_conditions('value')
task_layer.set_log_conditions('InputState-0')

colors_hidden_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('InputState-0')

words_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('InputState-0')



#   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
response_layer = pnl.RecurrentTransferMechanism(size=2,  #Recurrent
                         function=pnl.Logistic(),#pnl.Stability(matrix=np.matrix([[0.0, -1.0], [-1.0, 0.0]])),
                         auto=1,
                         name='RESPONSE',
                         output_states = [pnl.RECURRENT_OUTPUT.RESULT,
                                          {pnl.NAME: 'DECISION_ENERGY',
                                           pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                           pnl.FUNCTION: pnl.Stability(default_variable=np.array([[0.0, -1.0, 0.0]]),
                                                                       metric=pnl.ENERGY)}],
                                                                       # matrix=np.array([[0.0, -1.0, 0.0],
                                                                       #                  [-1.0, 0.0, 0.0]]))}],
                                                                       # matrix=np.array([[0.0, -1.0], [-1.0, 0.0]]))}],
                         integrator_mode=True,#)
                        # noise=pnl.NormalDist(mean=0.0, standard_dev=.01).function)
                         smoothing_factor=0.1)

response_layer.set_log_conditions('value')
# response_layer.set_log_conditions('DECISION_ENERGY')
#response_layer.set_log_conditions('gain')


color_input_weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
                                                     # name = 'color_input')

# color_input_weights = pnl.MappingProjection(np.array([[1.0, 0.0, 0.0],
#                                                       [0.0, 1.0, 0.0],
#                                                       [0.0, 0.0, 0.0]]),
#                                                      name = 'color_input')

word_input_weights = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0]])


# word_input_weights = pnl.MappingProjection(np.array([[1.0, 0.0, 0.0],
#                                 [0.0, 1.0, 0.0],
#                                 [0.0, 0.0, 0.0]]),
#                                            name= 'word input')

task_input_weights = np.array([[1.0, 0.0],
                                [0.0, 1.0]])
                                     # [0.0, 0.0],
                                     # [1.0, 1.0]])

# color_task_weights  = pnl.MappingProjection(np.array([[3.0, 3.0],
#                                 [0.0, 0.0],
#                                 [0.0, 0.0]]),
#                                             name='color_task')

color_task_weights  = np.array([[3.0, 3.0],
                                [0, 0],
                                [0, 0]])

task_color_weights  = np.array([[3.0, 3.0, 0.0],
                                [0, 0, 0]])

color_response_weights = np.array([[1.5, 0.0],
                                   [0.0, 1.5],
                                   [0.0, 0.0]])
# color_response_weights = pnl.MappingProjection(np.array([[1.5, 0.0],
#                                    [0.0, 1.5],
#                                    [0.0, 0.0]]),
#                                                name= 'color_response')

response_color_weights = np.array([[1.5, 0.0, 0.0],
                                   [0.0, 1.5, 0.0]])

word_task_weights = np.array([[0.0, 0.0],
                              [3.0, 3.0],
                              [0.0, 0.0]])

task_word_weights = np.array([[0.0, 0.0, 0.0],
                              [3.0, 3.0, 0.0]])

# word_task_weights = pnl.MappingProjection(np.array([[0.0, 0.0, 0.0],
#                                 [3.0, 3.0, 0.0]]),
#                                           name= 'word_task')

# task_word_weights   = np.array([[0.0, 0.0],
#                                 [3.0, 3.0]])

# word_response_weights  = pnl.MappingProjection(np.array([[2.5, 0.0],
#                                    [0.0, 2.5],
#                                    [0.0, 0.0]]),
#                                                name='word_response')
word_response_weights  = np.array([[2.5, 0.0],
                                   [0.0, 2.5],
                                   [0.0, 0.0]])

response_word_weights  = np.array([[2.5, 0.0, 0.0],
                                   [0.0, 2.5, 0.0]])
#   CREATE PATHWAYS
#   Words pathway

#   Colors pathway


color_response_process = pnl.Process(pathway=[colors_input_layer,
                                              color_input_weights,
                                              colors_hidden_layer,
                                              color_response_weights,
                                              response_layer],
                                     name='COLORS_RESPONSE_PROCESS')

color_control_process = pnl.Process(pathway=[colors_hidden_layer,
                                             color_task_weights,
                                             task_layer],
                                    name='COLORS_TASK_PROCESS')

word_response_process = pnl.Process(pathway=[words_input_layer,
                                             word_input_weights,
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
                                                   task_input_weights,
                                                   task_layer,
                                                   task_color_weights,
                                                   colors_hidden_layer])

task_word_response_process = pnl.Process(pathway=[task_input_layer,
                                                  task_layer,
                                                  task_word_weights,
                                                  words_hidden_layer])

response_color_task_process = pnl.Process(pathway=[response_layer,
                                                   response_color_weights,
                                                   colors_hidden_layer])

response_word_task_process = pnl.Process(pathway=[response_layer,
                                                  response_word_weights,
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

# CREATE THRESHOLF FUNCTION FOR INITIALIZATION
#
# def pass_initial_threshold(colors, thresh):
#     colors1 = colors_hidden_layer.output_states.values[0][0] #red response
#     # results2 = response_layer.output_states.values[0][1] #green response
#     print(colors1)
#     # print(results2)
#     if colors1  >= thresh:
#         return True
#     # for val in results1:
#     #     if val >= thresh:
#     #         return True
#     return False
# initial_accumulator_threshold = 0.3
#
# terminate_initialization = {
#    pnl.TimeScale.TRIAL: pnl.While(pass_initial_threshold, colors_hidden_layer, initial_accumulator_threshold)
# }


#   CREATE THRESHOLD FUNCTION
#first value of DDM's value is DECISION_VARIABLE
def pass_threshold(response, thresh):
    results1 = response_layer.output_states.values[0][0] #red response
    results2 = response_layer.output_states.values[0][1] #green response
    # print(results1)
    # print(results2)
    if results1  >= thresh or results2 >= thresh:
        return True
    # for val in results1:
    #     if val >= thresh:
    #         return True
    return False
accumulator_threshold = 0.8

terminate_trial = {
   pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, accumulator_threshold)
}

def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, CN, WR):

    trialdict = {
    colors_input_layer: [red_color, green_color, neutral_color],
    words_input_layer: [red_word, green_word, neutral_word],
    task_input_layer: [CN, WR]
    }
    return trialdict

# Define initialization trials separately
# input just task and run once so system asymptotes
WR_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 1)
CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 1, 0)

# Initialize System:
# colors_hidden_layer.integrator_mode = False
# words_hidden_layer.integrator_mode = False
# task_layer.integrator_mode = False
# response_layer.integrator_mode = False

# RUN SYSTEM INITIALIZATION:
# System_Conflict_Monitoring.run(inputs=CN_trial_initialize_input, termination_processing=terminate_initialization)
# colors_hidden_layer.reinitialize([[0, 0]])
# words_hidden_layer.reinitialize([[0, 0]])


print('colors_hidden_layer after initial trial: ', colors_hidden_layer.output_states.values)
print('words_hidden_layer after initial trial: ', words_hidden_layer.output_states.values)
print('response_layer after initial trial: ', response_layer.output_states.values)
print('task_layer after initial trial: ', task_layer.output_states.values)

# response_layer.integrator_mode = True
# colors_hidden_layer.integrator_mode = True
# words_hidden_layer.integrator_mode = True
# task_layer.integrator_mode = True

response_layer.reinitialize([[0, 0]])
print('response_layer after reinitialization trial: ', response_layer.output_states.values)

CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR
CN_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 0, 1) #red_color, green color, red_word, green word, CN, WR

ntrials = 40
System_Conflict_Monitoring.run(inputs=CN_incongruent_trial_input, num_trials=ntrials)# termination_processing=terminate_trial)
r1 = response_layer.log.nparray_dictionary()

# words_hidden_layer.reinitialize([[0, 0]])
# colors_hidden_layer.reinitialize([[0, 0]])
# task_layer.reinitialize([[0, 0]])
# response_layer.reinitialize([[0, 0]])


# System_Conflict_Monitoring.run(inputs=CN_congruent_trial_input, num_trials=400)#, termination_processing=terminate_trial)
r2 = response_layer.log.nparray_dictionary()
# rr1 = r1['DECISION_ENERGY']
# rr2 = r2['DECISION_ENERGY']
rr2 = r2['value']
rrr2 = rr2.reshape(40,2)
# response_layer.log.print_entries()

# words_hidden_layer.log.print_entries()
# colors_hidden_layer.log.print_entries()
#
# task_layer.log.print_entries()
#
# plt.plot(rr1)
plt.plot(rrr2)
plt.show()

# plt.figure()
# legend = ['control',
#         'incongruent',
#         'congruent']
# colors = ['b', 'g', 'r']
# t = np.arange(1.0, ntrials+1, 1.0)
# plt.plot(t,results_CN_control_trial[0:ntrials], 'b')
# plt.plot(t,results_CN_congruent_trial[0:ntrials],'r')
# plt.plot(t,results_CN_incongruent_trial[0:ntrials], 'g')
#
# plt.tick_params(axis='x', labelsize=9)
# plt.title('Conflict Monitoring')
# plt.legend(legend)
# plt.xlabel('trials')
# plt.ylabel('ENERGY')
# plt.show()

#
