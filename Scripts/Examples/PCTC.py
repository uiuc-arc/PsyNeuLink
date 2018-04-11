import numpy as np
import matplotlib.pyplot as plt
import psyneulink as pnl


#The main problem with bidirectional projections is that the task demand layer is not an origin mechanism !!!
# Is there a hack around it? Or an official way? Composition ??

# MODEL ARCHITECTURE
#
pc = 0.15 # proactive control: Values of pc used in The article range from 0.025 to 0.28 to fit empirical data
# The model parameters are presented in Table 1 of the article.
wc1 = 1   # connection strength from Control to Color units
wcw = 1   # connection strength from Control to Word units
wc2 = 2   # connection strength from Color to Control units
wwc = 2   # connection strength from Word to Control units

SettleTime = 500 #time until threshold should be reached


PC = np.array([[pc, 0.0]])# proactive control only biases the color task demand unit
TaskConflict_To_Response   = 500
ResponseThreshold          = 0.7
inh   = 1.3 # inhibition within color-, word-, and response-layers
inh_t = 1.9 # inhibition between task control units
bias = 0.3  # bias to input (word + color) units to prevent preresponding
Lambda = 0.97 # Euler integration constant


# SET UP MECHANISMS

#  INPUT UNITS

#  colors: ('red', 'green'), words: ('RED','GREEN')

color_word_input_layer = pnl.TransferMechanism(size=4,
                                               function=pnl.Linear,
                                           name='COLOR_WORD_INPUT')

color_feature_layer1 = pnl.RecurrentTransferMechanism(size=2,
                                                     integrator_mode=True,
                                                     function=pnl.Linear(),
                                                     auto=,
                                                     hetero=,
                                                     smoothing_factor=Lambda,     # integrate the lambda function from matlab here
                                           #function=pnl.Linear,
                                          name='COLOR_FEATURES')


word_feature_layer = pnl.RecurrentTransferMechanism(size=2,
                                                # integrate the lambda function from matlab here
                                           #function=pnl.Linear,
                                          name='WORD_FEATURES')


task_demand_layer = pnl.RecurrentTransferMechanism(size=2,
                                             # integrate the lambda function from matlab here
                                           #function=pnl.Linear,
                                           name='TASK_DEMAND')


response_layer = pnl.RecurrentTransferMechanism(size=2,
                                                # integrate the lambda function from matlab here
                                           #function=pnl.Linear,
                                                name='RESPONSE')


#   LOGGING

# Creating sub matrices, one between each of set of layers. A more elegant
# way is to treat the entire architecture as a (sparse) matrix with
# connections and then run the entire model in a single line. However, to
# keep it comprehensible, a step-by-step breakdown is provided here. It
# should still run fast, though.

color_word_input_weights = np.array([[1.0, 1.0],
                                     [0.0, 0.0],
                                     [0.0, 0.0],
                                     [1.0, 1.0]])

color_task_weights  = np.array([[wc2, 0.0],
                                   [0.0, wc2]])

task_color_weights  = np.array([[wc1, wc1],
                                [0, 0]])

color_response_weights = np.array([[2.0, 0.0],
                                   [0.0, 2.0]])


word_task_weights   = np.array([[0.0, wwc],
                                   [0.0, wwc]])

task_word_weights   = np.array([[0.0, 0.0],
                                [wcw, wcw]])

color_response_weights  = np.array([[2.0, 0.0],
                                    [0.0, 2.0]])

word_response_weights  = np.array([[2.5, 0.0],
                                   [0.0, 2.5]])

task_response_weights = np.array([[500.0, 0.0],
                                  [0.0, 500.0]])
#   CREATE PATHWAYS
#   Words pathway

#   Colors pathway


color_response_process = pnl.Process(pathway=[color_word_input_layer,
                                              color_word_input_weights,
                                              color_feature_layer,
                                              color_task_weights,
                                              color_response_weights,
                                              response_layer],
                                     name='COLORS_RESPONSE_PROCESS')

color_control_process = pnl.Process(pathway=[color_word_input_layer,
                                             color_word_input_weights,
                                             color_feature_layer,
                                             color_task_weights,
                                             task_demand_layer],
                                    name='COLORS_TASK_PROCESS')

word_response_process = pnl.Process(pathway=[color_word_input_layer,
                                             color_word_input_weights,
                                             word_feature_layer,
                                             word_response_weights,
                                             response_layer],
                                     name='COLORS_RESPONSE_PROCESS')

word_control_process = pnl.Process(pathway=[color_word_input_layer,
                                            color_word_input_weights,
                                            word_feature_layer,
                                            word_task_weights,
                                            task_demand_layer],
                                   name='WORDS_TASK_PROCESS')

# This does not work since task_demand_layer is not a origin mechanism
# task_response_process = pnl.Process(pathway=[task_demand_layer,
#                                              task_response_weights,
#                                              response_layer],
#                                    name='TASK_RESPONSE_PROCESS')
#
#
# task_color_response_process = pnl.Process(pathway=[task_demand_layer,
#                                                    task_color_weights,
#                                                    color_feature_layer,
#                                                    color_response_weights,
#                                                    response_layer])
#
task_word_response_process = pnl.Process(pathway=[task_demand_layer,
                                                   task_word_weights,
                                                   word_feature_layer,
                                                   word_response_weights,
                                                   response_layer])

#   CREATE SYSTEM
my_PC = pnl.System(processes=[color_control_process,
                              color_response_process,
                              word_control_process,
                              task_response_process,
                              word_response_process,
                              task_color_response_process,
                              task_word_response_process],
                   name = 'PROACTIVE_CONTROL_STROOP_SYSTEM')



my_PC.show()
my_PC.show_graph(show_dimensions=pnl.ALL)  # , output_fmt = 'jupyter')


# my_Stroop.show_graph(show_mechanism_structure=pnl.VALUES, output_fmt = 'jupyter')

# Function to create test trials
# a RED word input is [1,0] to words_input_layer and GREEN word is [0,1]
# a red color input is [1,0] to colors_input_layer and green color is [0,1]
# a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]

# def trial_dict(red_color, green_color, red_word, green_word, CN, WR):
    # trialdict = {
    #     colors_input_layer: [red_color, green_color],
    #     words_input_layer: [red_word, green_word],
    #     task_layer: [CN, WR]
    # }
    # return trialdict


# Define initialization trials separately

my_PC.run(inputs=[1,0,0,0])