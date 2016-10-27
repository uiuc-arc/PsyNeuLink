Search.setIndex({envversion:47,filenames:["AdaptiveIntegrator","Comparator","ControlMechanism","ControlSignal","DDM","DefaultControlMechanism","EVCMechanism","InputState","Learning","Log","Mapping","Mechanism","MonitoringMechanism","OutputState","ParameterState","Preferences","Process","ProcessingMechanism","Projection","Run","State","System","Transfer","Utilities","UtilityFunction","WeightedError","index"],objects:{"":{Comparator:[1,0,0,"-"],ControlMechanism:[2,0,0,"-"],ControlSignal:[3,0,0,"-"],DDM:[4,0,0,"-"],DefaultControlMechanism:[5,0,0,"-"],InputState:[7,0,0,"-"],Mapping:[10,0,0,"-"],Mechanism:[11,0,0,"-"],MonitoringMechanism:[12,0,0,"-"],OutputState:[13,0,0,"-"],ParameterState:[14,0,0,"-"],Preferences:[15,0,0,"-"],Process:[16,0,0,"-"],ProcessingMechanism:[17,0,0,"-"],Projection:[18,0,0,"-"],Run:[19,0,0,"-"],State:[20,0,0,"-"],System:[21,0,0,"-"],Transfer:[22,0,0,"-"],WeightedError:[25,0,0,"-"]},"Comparator.Comparator":{"__execute__":[1,2,1,""],instantiate_input_states:[1,2,1,""],terminate_function:[1,2,1,""]},"ControlMechanism.ControlMechanism_Base":{"__execute__":[2,2,1,""],Linear:[2,4,1,""],instantiate_control_mechanism_input_state:[2,2,1,""],instantiate_control_signal_projection:[2,2,1,""]},"ControlMechanism.ControlMechanism_Base.Linear":{"function":[2,2,1,""],derivative:[2,2,1,""]},"ControlSignal.ControlSignal":{compute_cost:[3,2,1,""],execute:[3,2,1,""],instantiate_receiver:[3,2,1,""],instantiate_sender:[3,2,1,""]},"ControlSignal.ControlSignalChannel":{"__getnewargs__":[3,2,1,""],"__new__":[3,5,1,""],"__repr__":[3,2,1,""],inputState:[3,1,1,""],outputIndex:[3,1,1,""],outputState:[3,1,1,""],outputValue:[3,1,1,""],variableIndex:[3,1,1,""],variableValue:[3,1,1,""]},"ControlSignal.ControlSignalValuesTuple":{"__getnewargs__":[3,2,1,""],"__new__":[3,5,1,""],"__repr__":[3,2,1,""],cost:[3,1,1,""],intensity:[3,1,1,""]},"DDM.DDM":{"__execute__":[4,2,1,""],ou_update:[4,2,1,""],terminate_function:[4,2,1,""]},"DefaultControlMechanism.ControlSignalChannel":{"__getnewargs__":[5,2,1,""],"__new__":[5,5,1,""],"__repr__":[5,2,1,""],inputState:[5,1,1,""],outputIndex:[5,1,1,""],outputState:[5,1,1,""],outputValue:[5,1,1,""],variableIndex:[5,1,1,""],variableValue:[5,1,1,""]},"DefaultControlMechanism.DefaultControlMechanism":{instantiate_control_signal_channel:[5,2,1,""],instantiate_control_signal_projection:[5,2,1,""],instantiate_input_states:[5,2,1,""]},"Mapping.Mapping":{execute:[10,2,1,""],instantiate_receiver:[10,2,1,""]},"Mechanism.MechanismList":{"__getitem__":[11,2,1,""],get_tuple_for_mech:[11,2,1,""],mechanisms:[11,1,1,""],names:[11,1,1,""],outputStateNames:[11,1,1,""],outputStateValues:[11,1,1,""],values:[11,1,1,""]},"Mechanism.MechanismTuple":{"__getnewargs__":[11,2,1,""],"__new__":[11,5,1,""],"__repr__":[11,2,1,""],mechanism:[11,1,1,""],params:[11,1,1,""],phase:[11,1,1,""]},"Mechanism.Mechanism_Base":{adjust_function:[11,2,1,""],execute:[11,2,1,""],instantiate_input_states:[11,2,1,""],run:[11,2,1,""],terminate_execute:[11,2,1,""]},"MonitoringMechanism.MonitoringMechanism_Base":{update_monitored_state_changed_attribute:[12,2,1,""]},"ParameterState.ParameterState":{update:[14,2,1,""]},"Process.ProcessList":{processNames:[16,1,1,""],processes:[16,1,1,""]},"Process.ProcessTuple":{"__getnewargs__":[16,2,1,""],"__new__":[16,5,1,""],"__repr__":[16,2,1,""],input:[16,1,1,""],process:[16,1,1,""]},"Process.Process_Base":{"_allMechanisms":[16,1,1,""],"_isControllerProcess":[16,1,1,""],"_mech_tuples":[16,1,1,""],"_monitoring_mech_tuples":[16,1,1,""],"_origin_mech_tuples":[16,1,1,""],"_phaseSpecMax":[16,1,1,""],"_terminal_mech_tuples":[16,1,1,""],clamp_input:[16,1,1,""],execute:[16,2,1,""],input:[16,1,1,""],inputValue:[16,1,1,""],mechanismNames:[16,1,1,""],mechanisms:[16,1,1,""],monitoringMechanisms:[16,1,1,""],name:[16,1,1,""],numPhases:[16,1,1,""],originMechanisms:[16,1,1,""],outputState:[16,1,1,""],pathway:[16,1,1,""],prefs:[16,1,1,""],processInputStates:[16,1,1,""],results:[16,1,1,""],run:[16,2,1,""],systems:[16,1,1,""],terminalMechanisms:[16,1,1,""],timeScale:[16,1,1,""],value:[16,1,1,""]},"Projection.Projection_Base":{instantiate_receiver:[18,2,1,""],instantiate_sender:[18,2,1,""]},"State.State_Base":{check_projection_receiver:[20,2,1,""],check_projection_sender:[20,2,1,""],instantiate_projection_from_state:[20,2,1,""],instantiate_projections_to_state:[20,2,1,""],parse_projection_ref:[20,2,1,""],update:[20,2,1,""]},"System.System_Base":{"_allMechanisms":[21,1,1,""],"_all_mech_tuples":[21,1,1,""],"_control_mech_tuple":[21,1,1,""],"_learning_mech_tuples":[21,1,1,""],"_monitoring_mech_tuples":[21,1,1,""],"_origin_mech_tuples":[21,1,1,""],"_processList":[21,1,1,""],"_terminal_mech_tuples":[21,1,1,""],InspectOptions:[21,4,1,""],controlMechanisms:[21,1,1,""],execute:[21,2,1,""],executionGraph:[21,1,1,""],executionList:[21,1,1,""],execution_graph_mechs:[21,1,1,""],execution_sets:[21,1,1,""],graph:[21,1,1,""],inputValue:[21,1,1,""],inspect:[21,2,1,""],mechanisms:[21,1,1,""],mechanismsDict:[21,1,1,""],monitoringMechanisms:[21,1,1,""],numPhases:[21,1,1,""],originMechanisms:[21,1,1,""],processes:[21,1,1,""],run:[21,2,1,""],show:[21,2,1,""],terminalMechanisms:[21,1,1,""],value:[21,1,1,""]},"Transfer.Transfer":{"__execute__":[22,2,1,""],terminate_function:[22,2,1,""]},"WeightedError.WeightedError":{"__execute__":[25,2,1,""]},Comparator:{Comparator:[1,4,1,""],random:[1,3,1,""]},ControlMechanism:{ControlMechanism_Base:[2,4,1,""],random:[2,3,1,""]},ControlSignal:{ControlSignal:[3,4,1,""],ControlSignalChannel:[3,4,1,""],ControlSignalValuesTuple:[3,4,1,""],random:[3,3,1,""]},DDM:{DDM:[4,4,1,""],random:[4,3,1,""]},DefaultControlMechanism:{ControlSignalChannel:[5,4,1,""],DefaultControlMechanism:[5,4,1,""],random:[5,3,1,""]},InputState:{InputState:[7,4,1,""],instantiate_input_states:[7,3,1,""],random:[7,3,1,""]},Mapping:{Mapping:[10,4,1,""],random:[10,3,1,""]},Mechanism:{MechanismList:[11,4,1,""],MechanismTuple:[11,4,1,""],Mechanism_Base:[11,4,1,""],mechanism:[11,3,1,""],random:[11,3,1,""]},MonitoringMechanism:{MonitoringMechanism_Base:[12,4,1,""],random:[12,3,1,""]},OutputState:{OutputState:[13,4,1,""],instantiate_output_states:[13,3,1,""],random:[13,3,1,""]},ParameterState:{ParameterState:[14,4,1,""],instantiate_parameter_states:[14,3,1,""],random:[14,3,1,""]},Process:{ProcessInputState:[16,4,1,""],ProcessList:[16,4,1,""],ProcessTuple:[16,4,1,""],Process_Base:[16,4,1,""],process:[16,3,1,""],random:[16,3,1,""]},ProcessingMechanism:{ProcessingMechanism_Base:[17,4,1,""],random:[17,3,1,""]},Projection:{Projection_Base:[18,4,1,""],add_projection_from:[18,3,1,""],add_projection_to:[18,3,1,""],is_projection_spec:[18,3,1,""],is_projection_subclass:[18,3,1,""],random:[18,3,1,""]},Run:{random:[19,3,1,""],run:[19,3,1,""]},State:{State_Base:[20,4,1,""],check_parameter_state_value:[20,3,1,""],check_state_ownership:[20,3,1,""],instantiate_state:[20,3,1,""],instantiate_state_list:[20,3,1,""],random:[20,3,1,""]},System:{System_Base:[21,4,1,""],random:[21,3,1,""],system:[21,3,1,""]},Transfer:{Transfer:[22,4,1,""],random:[22,3,1,""]},WeightedError:{WeightedError:[25,4,1,""],random:[25,3,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","attribute","Python attribute"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","class","Python class"],"5":["py","staticmethod","Python static method"]},objtypes:{"0":"py:module","1":"py:attribute","2":"py:method","3":"py:function","4":"py:class","5":"py:staticmethod"},terms:{"0x106b6b5f8":1,"0x10b667f28":[1,2,3,4,5,7,10,13,14,16,21,22,25],"0x10b77e840":2,"0x10bbfd0f0":19,"0x10bbfd668":19,"0x10bbfde80":19,"0x10bbfdef0":19,"0x10bfa0cf8":19,"0x10bfa0da0":19,"0x10bfa0e48":19,"0x10bfdaef0":21,"0x10c1f48d0":19,"0x10c1f4c50":19,"0x10c5b6320":16,"0x10c699908":16,"0x10c6999e8":16,"0x10c699ba8":16,"0x10cd1ba20":21,"2afc":4,"__execute__":[1,2,4,22,25],"__getitem__":11,"__getnewargs__":[3,5,11,16],"__init__":[1,2,3,4,10,11,16,18,20,21,22,25],"__new__":[3,5,11,16],"__repr__":[3,5,11,16],"_all_mech_tupl":21,"_allmechan":[16,21],"_assign_input":11,"_cl":[3,5,11,16],"_control_mech_tupl":21,"_instantiate_attributes_after_funct":2,"_instantiate_attributes_before_funct":2,"_instantiate_funct":[1,4,7,14,18,22],"_instantiate_graph":16,"_iscontrollerprocess":16,"_learning_mech_tupl":21,"_mech_tupl":16,"_monitoring_mech_tupl":[16,21],"_origin_mech_tupl":[16,21],"_phasespecmax":[16,21],"_processlist":21,"_report_mechanism_execut":[11,22],"_terminal_mech_tupl":[16,21],"_update_output_st":11,"_validate_param":[2,7,11,13,14,18,25],"_validate_vari":[7,10,11,13,14],"abstract":[2,11,12,16,17,18,20,21],"case":[3,4,10,11,16,19,20,21],"class":[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,21,22,25],"default":[1,2,3,4],"enum":[1,2,3,11,21],"final":21,"float":[4,11,22],"function":[1,2,3,4,5,7,10],"import":11,"int":[2,11,16,21],"long":[7,10,11,14],"new":[2,3,5,11,16,20],"return":[1,2,3,4,5,7,11,12,13,14,16,18,19,20,21,22,25],"static":[3,5,11,16],"super":[7,10,11,14,18],"true":[2,7,10,11,14,16,18,19,21],"while":[11,18,19,20,21],about:5,abov:[1,4,10,11,16,18,19,20,21,22,25],absent:[11,20],access:[11,16,20,21],accommod:[2,3],accomod:2,accordingli:12,accur:4,achiev:19,across:[11,18,19,20,22],activ:22,activation_mean:22,activation_vari:22,acycl:21,add:[2,5,10,18],add_project:18,add_projection_from:18,add_projection_to:18,add_to:18,addit:[1,4,7,11,13,18,20,21,22,25],adjac:16,adjust:[1,3,4,7,10,11,13,14,22,25],adjust_funct:11,adjustment_cost:3,adjustmentcost:3,adjustmentcostfunct:3,affect:19,after:[11,16,19,21],again:16,against:16,aggreg:[11,14,19],aggregr:11,algorithm:16,alia:[3,5,11,16],all:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],all_output:21,all_output_label:21,all_output_st:[2,11,21],alloc:[3,18],allocation_sampl:3,allocationpolici:2,allocationsampl:3,allow:[3,10,11,14,18,19,20],alon:[2,16,19],along:[3,11,16],alreadi:18,also:[1,4,11,14,16,19,21,22,25],altern:[4,16],although:[7,14],alwai:[16,19],ambigu:16,among:[16,19],analysi:[4,19,21],analyt:[4,19,22],ani:[1,2,4,11,14,16,18,19,20,21,22,25],anoth:[7,10,11,14,16,21],anyth:14,appear:[2,16,19,21],append:19,appli:[2,11,16,21,22],appropri:[16,18,19,21],approxim:19,arg:[1,3,4,10,11,16,18,20,22,25],argument:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],arithmet:[7,14],arrai:[1,2,3,4,7,10,11,16,19,20,21,22,25],arri:2,assign:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],assign_default:[1,4,11,22,25],assign_funct:3,associ:[2,3,11,18,21,25],assum:[11,18,20],attent:4,attribut:[1,2,3,4,5,7,10,11,12,13,14,16,18,19,20,21,22,25],augment:[5,10],auto:3,autoassignmatrix:16,autom:3,automat:[1,3,7,10,13,14,16,19],autonumb:[2,11],avail:16,averag:22,axi:[16,19,21],backpropag:16,bai:4,ballist:19,base:[1,2,3,4,10,11,16,18,19,20,21,22,25],basevalu:[14,20],becaus:19,been:[11,13,16,18,19,21],befor:[3,11,16,19,21],begin:21,behavior:[3,16],belong:[11,16,18,20,21],below:[3,4,11,16,18,19,21],best:19,between:[4,16,19,21],bia:[4,11,22],bogacz:4,bogaczet:4,bool:[11,16,18,19,21],both:[10,11,16,19,20,21],brown:4,calcul:[2,4,7,19,25],call:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],call_after_execut:11,call_after_time_step:[16,19,21],call_after_tri:[16,19,21],call_before_execut:11,call_before_time_step:[16,19,21],call_before_tri:[16,19,21],caller:20,can:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],cap:22,categori:[1,3,4,7,10,11,13,14,18,20,21,22,25],centralclock:[4,16,19,21],chain:21,chang:[1,3,4,7,10,12,13,14,22,25],channel:3,check:[3,10,14,16,18,20],check_parameter_state_valu:20,check_projection_receiv:20,check_projection_send:20,check_state_ownership:20,choic:4,clamp_input:16,clariti:[11,19],classnam:[10,11,18,20],classprefer:[1,3,4,10,11,14,16,18,20,21,22,25],classpreferencelevel:[1,3,4,10,11,14,18,20,22,25],close:[19,21],closest:19,cohen:4,collect:21,color:3,column:25,combin:[1,2,3,4,7,14,16,19,20,22,25],combinationoper:10,commit:19,common:19,comparator_default_starting_point:1,comparator_preferenceset:1,comparator_sampl:1,comparator_target:1,comparatormechan:19,comparatoroutput:1,comparatorsampl:1,comparatortarget:1,comparis:1,comparison:[1,25],comparison_arrai:1,comparison_mean:1,comparison_oper:1,comparison_sum:1,comparison_sum_squar:1,comparison_typ:1,comparisonfunct:1,compat:[7,11,14,16,18,20,21],compati:[16,20],complet:[7,13],compon:[4,11],comput:[1,2,3,4,11,20,22,25],compute_cost:3,concaten:11,concept:19,condit:[19,20],confid:[1,4,11,22],configur:[3,16],confirm:[4,7,10,14,22],conform:[7,14],connect:16,connectionist:16,consist:[11,16,19],constant:[18,20],constrain:21,constraint:[5,11,14,18,20],constraint_valu:[7,20],constraint_value_nam:20,construct:[10,11,16,20,21],constsant:22,contain:[4,7,11,13,16,18,19,20,21,22],context:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,21,22,25],contextu:16,continu:[16,19],contribut:[3,25],control_function_typ:3,control_mechan:21,control_projection_receiv:21,control_sign:14,control_signal_alloc:11,controlmechan:[2,11,16,21],controlmechanism_bas:2,controlmodulatedparamvalu:[1,4,11,22],controlproject:21,controlsign:[1,2,3,4,5,11,14,18,22,25],controlsignalajdustmentcostfunct:3,controlsignalchannel:[3,5,18],controlsignalcost:[2,3],controlsignaldurationcostfunct:3,controlsignalintensitycostfunct:3,controlsignallog:3,controlsignalpreferenceset:3,controlsignaltotalcostfunct:3,controlsignalvaluestupl:3,controlst:18,convei:[10,18],conveni:11,convent:[16,19,21],convert:[3,4,16,19],coordin:[11,16],copi:[1,3,4,5,10,11,16,18,20,22,25],core:11,correct:[3,4,10,11,19],corresond:10,correspond:[1,3,4,7,11,13,16,19,20,21,22,25],corrrespond:21,cost:[2,3,5],costfunctionnam:3,could:18,count:[1,3,4,7,10,11,13,14,18,20,22,25],creat:[1,3,4,5,7,10,11,13,14,16,18,20,21,22,25],criterion:19,curent:3,currenlti:16,current:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],current_monitored_st:12,currentstatetupl:[1,4,11,22],cycl:21,ddm_analyt:4,ddm_default:4,ddm_distr:4,ddm_preferenceset:4,ddm_rt:4,ddm_updat:4,deal:16,decai:4,decion:4,decis:4,decision_vari:4,dedic:3,defaul:2,default_alloc:3,default_allocation_sampl:3,default_input_valu:[2,4,5,11,16,19,21,22],default_matrix:10,default_projection_matrix:16,default_sample_and_target:1,default_sample_valu:3,defaultcontrol:[2,3,5,11,21],defaultcontrolalloc:[3,5],defaultcontrolmechan:[5,11,18,21],defaultmechan:[11,16],defaultprocess:21,defaultprocessingmechanism_bas:[11,14],defaultreceiv:10,defaultsend:10,defin:[2,16,19,21],definit:[14,16,18],delet:[1,2,4,22],depend:[19,21],deriv:[2,21,25],describ:4,descript:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],descriptor:11,design:[3,11,16,19,21],desir:16,destin:10,detail:[7,10,11,13,16,19,20,21],determin:[1,2,3,4,11,16,18,19,21,22],deviat:22,devoid:21,diciontari:21,dict:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],dict_output:21,dictionari:[1,3,4,7,10,11,13,14,16,18,19,20,21,22,25],differ:[1,4,11,16,18,20,21,22,25],differec:1,diffus:4,dimens:19,dimension:19,direct:[3,11,16,18,20],directi:19,directli:[1,3,4,7,10,11,13,14,16,19,22,25],disabl:[3,16,19,21],discuss:19,distribut:4,divis:1,document:[2,4,5,14],doe:[5,11,16,21],don:21,done:[5,10,19],drift:4,drift_rat:4,dtype:22,duplic:[16,21],durat:[1,3,4,11,22],durationcost:3,durationcostfunct:3,dure:[16,19,21],each:[2,3,4,5,7,11,13,14,16,18,19,20,21,22,25],easier:19,effect:[7,10,13,14,16,18,21],effienc:3,either:[3,4,10,11,16,18,19,21],element:[1,3,11,19,25],elementwis:11,elig:16,els:20,emb:19,embed:11,emo:11,emp:11,empti:[20,21],emv:11,enabl:[3,16,19,21],enable_control:21,encod:16,enforc:[7,13,14,18],engin:4,enrti:11,enter:16,entir:16,entri:[1,2,3,4,7],equal:[1,16,19,21,25],error:[4,10,11,16,20],error_arrai:25,error_r:4,error_sign:25,esp:21,establish:[11,20],estim:4,etc:[4,5,16],evalu:[3,11,16,18],evc:[2,3,5,11],even:[16,18,19],ever:19,everi:[4,5,11,16,19,21],exampl:[11,19,21],except:[3,11,18,19,20],execut:[1,2,3,4,5,7,10,11,13,14,16,18,19,20],executeparamspec:11,execution_graph_mech:21,execution_set:21,executiongraph:[2,21],executionlist:[19,21],exist:16,expect:11,explain:5,explan:16,explant:16,explic:10,explicit:[3,7,10,13,14],explicitli:[1,2,3,4,7,10,11,13,14,16,18,20,21,22,25],expon:[2,11,21],exponenti:[2,3,11,21,22],extend:2,extrem:19,factor:19,factori:[11,16,21],fail:20,fall:21,fals:[11,14,16,18,19,20,21],fast:4,few:19,field:[3,5,11,16],figur:[11,16,19],first:[2,4,7,10,11,13,16,19,20,22],fix:10,flag:11,flat_output:21,float64:22,follow:[1,2,4,11,16,18,19,20,21,22,25],forc:[4,16,19,21],form:[4,11,14,16,19,20],formal:4,format:[3,5,11,16,19,21],forth:19,four:19,framework:[16,19,21],from:[1,2,3,4,7,10,11,14,16,18,20,21,22,25],full:[7,11,13,18],full_connectivity_matrix:16,fulli:19,function_nam:3,function_param:[1,2,3,4,7,10,11,13,14,20,22],function_run_time_parm:[1,4,22,25],functioncategori:[11,18,20],functionnam:3,functionparam:1,functionparrameterst:10,functionpreferenceset:[16,21],functiontyp:[1,2,3,4,5,7,10,11,13,14,20,22,25],fundament:11,further:19,fuss:4,gain:[11,22],gate:[11,18],gaussian:22,gener:[1,3,4,11,14,16,19,20,25],get:[7,11,14,18,20],get_adjust:3,get_cost:3,get_duration_cost:3,get_ignoreintensityfunct:3,get_intensity_cost:3,get_tuple_for_mech:11,give:4,given:[16,19,20],granular:[4,22],graph:[19,21],hadamard:[1,11],handl:[1,3,4,10,11,19,20,22,25],hard_clamp:16,have:[3,10,11,13,16,18,19,20,21],help:19,here:[1,3,13,14,18],hiearchic:21,hierarch:16,higher:19,histori:3,holm:4,how:[2,3,16,19,20],howev:[11,16,19,21],hyphen:[1,3,4,7,10,11,13,14,18,20,22,25],ident:[3,11,13],identifi:[3,16,20],identity_matrix:[10,13,16],identitymap:[3,10],identitymatrix:10,ignor:[3,11,16,19,20,21],ignoreintensityfunct:3,implement:[1,3,4,5,7,10,11,13,14,16,18,19,20,22,25],implementt:10,includ:[3,7,10,11,13,14,16,18,20,21],incom:[11,18],increment:[11,20],index:[1,3,4,7,10,11,13,14,16,18,20,21,22,25,26],indic:[16,22],individu:[7,21],infer:[14,18],inherit:14,init:[7,10,11,13,14,18],initalize_cycl:21,initi:[1,2,3,4,7,10,11,13,14,16,18],initial_cycl:21,initial_st:22,initial_valu:[16,19,21],initialize_cycl:[19,21],initialize_cyl:11,inlin:[11,16],inner:19,input:[1,2,3,4,5],input_arrai:21,input_st:2,input_state_nam:2,input_state_param:11,input_state_valu:2,input_templ:11,inputst:[1,2,3,5,7,10,11,16,18,19,20,25],inputstateparam:7,inputvalu:[4,11,16,21,22],insid:19,inspect:21,inspectionopt:21,inspectopt:21,instal:4,instanc:[1,2,3,4,5,7,10,11,13,14,16,18,20,21,22,25],instancecount:11,instanti:[1,2,3,4,5,7,10,11,13,14,16,18,20,21,22,25],instantiate_control_mechanism_input_st:2,instantiate_control_signal_channel:5,instantiate_control_signal_project:[2,5],instantiate_input_st:[1,5,7,11],instantiate_output_st:13,instantiate_parameter_st:14,instantiate_projection_from_st:20,instantiate_projections_to_st:[18,20],instantiate_receiv:[3,10,18],instantiate_send:[3,10,18],instantiate_st:[11,20],instantiate_state_list:[7,13,14,20],instead:20,insur:[3,7,14,18,19],intact:[16,19,21],intantiate_mechanism_st:20,intens:[2,3],intensity_cost:3,intensitycost:3,intensitycostfunct:3,intercept:[2,3,11],interfac:3,intern:[4,11,16,19,21],interpos:16,interv:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,19,20,21,22,25],intial_valu:19,intiti:4,intreg:4,invalid:20,invok:[11,19],involv:16,is_pref_set:[1,2,3,4,5,7,10,13,14,16,21,22,25],is_projection_spec:18,is_projection_subclass:18,isn:[3,10],item:[1,2,3,7,11,13,16,19,20,21,22],iter:19,itself:[2,11,21],journal:4,just:[2,7,10,11,13,14,20],karg:20,keep:[5,11],kei:[3,7,10,11,13,14,16,18,19,20,21],kept:[1,4,22,25],keypath:[10,11,20],keyword:[3,11,16,18,20,21],know:5,kvo:20,kwaddinputst:18,kwaddoutputst:18,kwallocationsampl:3,kwaritment:[7,14],kwbogaczet:4,kwcontextarg:11,kwcontrolsignaladjustmentcostfunct:3,kwcontrolsignalcost:3,kwcontrolsignalcostfunct:3,kwcontrolsignalcostfunctionnam:3,kwcontrolsignaldefaultnam:3,kwcontrolsignaldurationcostfunct:3,kwcontrolsignalident:3,kwcontrolsignalintensitycostfunct:3,kwcontrolsignallogprofil:3,kwcontrolsignalparam:11,kwcontrolsignaltotalcostfunct:3,kwddm:11,kwddm_analyticsolut:4,kwddm_bia:4,kwdefaultsend:18,kwexecut:16,kwexponenti:22,kwfunctionoutputstatevaluemap:11,kwinputst:[7,10,11,13,18,20],kwinputstatevalu:[2,5],kwintegr:4,kwlearn:20,kwlinear:22,kwlinearcombinationfunct:[7,14],kwlinearcombinationoper:10,kwlogist:22,kwmappingfunct:10,kwmappingparam:11,kwmechan:20,kwmechanismexecutionsequencetempl:11,kwmechanismfunctioncategori:11,kwmechanismtimescal:11,kwmechanismtyp:11,kwmechanisparameterst:14,kwmonitoredst:2,kwmstateproject:20,kwnamearg:11,kwnavarroandfuss:4,kwnavarrosandfuss:4,kwoutputst:[4,7,11,13,18,20],kwoutputstatevalu:[2,5],kwparameterst:14,kwparammodulationoper:14,kwprefsarg:11,kwprojectionfunctioncategori:18,kwprojectionparam:[11,18,20],kwprojectionsend:[3,10,18],kwprojectionsendervalu:[3,10,18],kwreceiv:10,kwstate:20,kwstatefunctioncategori:20,kwstateparam:20,kwstateprojectionaggregationfunct:7,kwstateprojectionaggregationmod:7,kwstatevalu:20,kwtimescal:[1,4,22],kwtransfer_output:22,kwvalid:11,label:21,lambda:[10,11,14],largest:16,last:[3,13,16,19,21],last_alloc:3,last_intens:3,later:[1,4,22,25],latter:[1,4,19,22,25],layer:[16,25],learnin:16,learning_projection_receiv:21,learning_sign:10,learningmechan:21,learningsign:[16,18,20,25],least:[19,25],left:[11,16,19,21],legal:18,len:11,length:[1,11,16,19,20,21,25],lengthen:3,level:[4,16,19,21,22],like:[11,19],linear:[2,3,5,11,22],linearcombin:[1,3,7,11,13,14,20],linearmatrix:[10,13],link:[11,16,19,21],list:[2,3,7,10,11,13,14,16,18,19,20,21],local:1,log:3,log_all_entri:3,log_profil:3,logist:[11,22],logprofil:3,loop:[16,19,21],lowest:[16,19,21],made:11,mai:[5,7,14,16,18],maintain:[1,3,4,7,10,11,13,14,18,20,22,25],make:[4,16,20],make_default_control:2,manag:19,map:[3,5,7],mappingpreferenceset:10,match:[11,16,19,20,25],mathemat:4,matlab:4,matrix:[10,13,16,25],maximum:[21,22],mean:[1,4,22],mech:[11,21],mech_spec:11,mech_tupl:[11,21],mechainsm:11,mecham:2,mechan:1,mechanim:[18,19],mechanism_bas:11,mechanism_basepreferenceset:11,mechanism_st:20,mechanisminput:11,mechanismlist:[11,16,21],mechanismnam:16,mechanismouput:11,mechanismregistri:[1,4,11,22,25],mechanismsdict:21,mechanismsinputst:1,mechanismsparameterst:20,mechanismtupl:[11,16],mechansim:[20,21],member:16,messag:[16,20],met:19,method:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],michael:4,mind:19,minimum:22,miss:16,mode:[19,21],model:4,modifi:[1,3,4,11,16,18,21,22,25],modul:[3,10,11,16,18,21,26],modulationoper:[10,14],moehli:4,monitor:[2,11],monitored_output:21,monitored_output_label:21,monitored_output_st:[2,11,21],monitored_outputst:21,monitoredoutputst:2,monitoredoutputstatesopt:[2,11,21],monitoredstatechang:12,monitoring_mechan:21,monitoringmechan:[1,10,11,12,16,18,19,21,25],monitoringmechanism_bas:12,more:[5,7,10,11,13,14,16,19,20,21],most:[3,19],move:2,much:19,multi:[11,20],multipl:[2,7,10,11,14,16,18,19,20,21],multipli:[14,22],must:[1,2,3,4,7,10,11,14,16,18,19,20,21,22,25],my_mechan:11,name:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,21,22,25],navarro:4,navarroandfuss:4,navarrosandfuss:4,ndarrai:[11,16,19,20,21],necessari:[11,20],necessarili:[11,16,21],need:[2,4,5,16,18],nest:[16,19,21],network:16,never:[11,18,20],nevertheless:19,next:[11,16,19,21,25],next_level_project:25,nice:[3,5,11,16],nois:[4,22],non:[4,11],non_decision_tim:4,none:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,19,20,21,22,25],notat:11,note:[1,2,3,4,7,10,11,13,14,16,18,19,20,22,25],notimpl:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,22,25],nth:11,num_execut:[11,16,19,21],num_phases_per_tri:21,number:[2,3,4,5,10,11,16,19,20,21,22,25],numer:[11,19],numphas:[16,21],object:[1,2,3,4,7,11,13,14,16,18,19,20,21,22],observ:20,obviou:[18,20],occur:[11,21],off:[16,19,21],offer:[11,20],omit:[1,3,4,7,10,11,13,14,16,18,20,22,25],onc:[16,19],oni:4,onli:[2,3,4,7,11,13,16,18,19,20,21,22],oper:[1,7,11,13,14,18],optim:4,option:[1,2,3,4,11,16,18,19,20,21,22,25],order:[1,4,11,16,19],ordereddict:[7,11,13,18,20,21],ordereddictionari:20,organ:21,origin:[11,16,19,21],origin_mechan:21,originmechan:[16,21],other:[2,3,4,10,11,14,16,19,21,25],otherwis:[3,10,11,16,18,19,20],ou_upd:4,ouputst:5,out:20,outcom:[1,4,11,22],outermost:[16,19,21],outgo:18,output:[1,2,3,4,5,7,10,11],output_state_nam:21,output_state_param:11,output_value_arrai:21,outputindex:[3,5],outputst:[1,2,3,4,5,10,11,13,16,18,19,20,21,22,25],outputstatenam:11,outputstatevalu:11,outputstatevaluemap:11,outputvalu:[3,5,11,21],outsid:21,outstat:[2,11,21],over:16,overrid:[2,3,5,10,11,16,18,21],overridden:[1,2,4,11,18,21,22,25],overriden:2,own:[10,11,16],owner:[3,7,11,13,14,16,18,20],page:26,param:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,21,22,25],param_default:[1,4,22,25],param_modulation_oper:10,param_nam:20,paramclassdefault:[1,2,3,4,5,7,10,11,13,14,18,20,22,25],paramet:[1,2,3,4,5,7,10,11,12,13],parameter:21,parameter_modulation_oper:14,parameter_spec:2,parameter_state_param:[10,11,14],parameterst:[1,3,4,10,11,14,18,20,21,22,25],parameterstateparam:14,paraminstancedefault:[1,4,7,10,11,13,14,18,22,25],parammodulationoper:14,paramnam:[1,3,4,7,10,11,13,14,18,20,22,25],params_dict:20,paramscurr:[1,4,7,10,11,13,14,18,20,22,25],paramvalueproject:[11,14,20],pars:[7,11,14,19],parse_projection_ref:20,part:[3,7,10,11,13,14,16,18,20],particl:4,particular:[16,18],pass:[3,4,5,7,10,11,14,18,20,21,22],passag:4,pathwai:[7,10,11,13,16,18,19,21],per:[4,19],perform:[4,11],permiss:[11,21],permit:21,phase:[11,16],phasespec:[11,16,19,21],physic:4,pickl:[3,5,11,16],place:[11,16,20],plain:[3,5,11,16],plot:2,plural:11,point:[11,21],popul:18,possibl:[10,19],preced:[7,10,16],pref:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,21,22,25],prefer:[10,11],preferenceentri:[10,11,20],preferencelevel:[1,3,4,10,11,14,18,20,22,25],preferenceset:[1,3,4,10,11,16,18,20,21,22,25],prefix:20,present:[11,19],preserv:16,primari:[2,10,11,16,18,19,21],primarili:13,primary_output:21,primary_output_label:21,primary_output_st:[2,11,21],print:[2,21],prior:19,probability_lower_bound:4,probability_upper_bound:4,process:[1,4,7,10],process_bas:16,process_input:16,process_spec:16,processingmechan:[11,16,17,21,22,25],processingmechanism_bas:17,processinputst:[11,16],processlist:[16,21],processnam:16,processregistri:16,processtupl:16,produc:[7,14],product:[11,13,14],project:[2,3,5,7,10,11,13,14],projection_bas:18,projection_spec:[18,20],projection_typ:[14,18,20],projectionnam:11,projectionpreferenceset:18,projectionregistri:[3,10,18],projectionsend:18,projectoin:18,properti:11,proport:22,protocol:2,provid:[1,3,4,11,16,18,19,20,21,22,25],prune:21,psycholog:4,psyneulink:[3,4,7,11,13,14,16,19,21,22],psyneulnk:11,purpos:[11,19],put:3,python:11,question:1,quotient:1,rais:[3,11,18,20],random:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,19,20,21,22,25],random_connectivity_matrix:16,rang:22,rate:[4,22],rather:[3,10,11,13,18,20],reaction:4,read:21,real:[19,21],real_tim:4,reassign:[10,20],receiv:[3,7,10,11,14,16,18,20,21,25],receivesfromproject:[11,18,20],receivesprocessinput:11,recent:3,reciev:18,record:11,recurr:[16,21],recurrent_init_arrai:21,recurrent_mechan:21,recurrentinitmechan:21,redund:10,ref:[2,3,10,11,20],refer:[1,11,14,16,18,20],referenc:21,reference_valu:[7,13,14],referenceto:[2,11],regist:[1,3,4,7,10,11,13,14,18,20,22,25],register_categori:[1,3,4,10,11,18,20,22,25],registri:[11,16,18,21],reinforc:16,relev:[11,19],remain:21,replac:18,report:3,repres:[7,11,13,14,20,21],represent:[3,5,11,16],request:[2,3,18,20],request_set:[2,11,25],requir:[3,4,7,10,11,13,14,16,19,21],requiredparamclassdefaulttyp:[18,20],reset:[11,16,19,21],reset_clock:[16,19,21],resolv:11,respect:21,respons:[4,11,19],resposn:3,restrict:[1,4,22],result:[2,11,14,16,19,20,21,22],review:4,round:[16,19,21],rouund:19,row:25,rt_correct_mean:4,rt_correct_vari:4,rt_mean:4,rtype:[1,3,4,11,22],run:[4,7,10,11,13,14,16],runtim:[1,4,11,14,16,22,25],runtime_param:[2,4,11,16,21,22],same:[1,7,10,11,13,16,18,19,20,21,25],sampl:[1,3,11,21],sample_input:19,scalar:[2,3,19],scale:[11,16,19],schemat:11,scope:19,search:[3,10,26],second:[11,16,20,22],see:[1,3,4,7,10,11,13,16,18,19,20,21,22,25],select:11,self:[1,2,3,4,5,7,10,11,12,13,14,16,18,20,21,22,25],send:[13,16,18,21],sender:[2,3,5,7,10,11,13,14,16,18,19,20,21,25],senderdefault:10,sendstoproject:[18,20],separ:[1,4,22,25],sequenc:[11,16,19,21],sequeunc:19,serv:[11,18,21],set:[1,3,4,7,10,11,12,13,14,16,18,19,20,21,22,25],set_adjustment_cost:3,set_allocation_sampl:3,set_duration_cost:3,set_intensity_cost:3,set_log:3,set_log_profil:3,set_valu:20,sever:[1,3,4,10,11,21,22,25],share:11,should:[3,5,10,11,13,16,18,19,20,21],show:[2,11,19,21],shown:11,shvartsman:4,sigmoid:2,signal:2,similarli:11,simpler:19,simplest:19,simpli:[3,5,11,13,19],simul:4,sinc:[18,20,21],singl:[2,3,4,7,10,11,14,16,18,19,20,21],singleton:[11,21],singular:11,situat:[3,10],size:19,slope:[2,3,11],soft_clamp:16,softmax:2,sole:[11,20],solut:[4,22],some:[11,16,19,21],someth:5,sophist:5,sort:21,sourc:[3,5,10,16,19],spec:[11,16,18,20,21],special:4,specif:[2,3,7,10,11,13,14,16,18,19,20,21],specifi:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],squar:[1,25],squqr:1,stand:[16,19],standard:[1,4,11,18,20,22,25],start:[16,21],starting_point:4,state:[1,2,3,4,5],state_bas:[7,13,20],state_list:20,state_nam:20,state_param:20,state_param_identifi:20,state_project:[11,20],state_spec:[2,20],state_typ:20,statepreferenceset:20,stateregistri:[7,11,13,14,18,20],statist:[1,25],std:[1,22,25],step:[4,11,16,19,21],stepwis:4,still:[3,10,11,16,18,20],stochast:4,stop:4,store:[11,16,19,21],str:[1,2,3,4,5,7,10,11,13,14,16,18,20,21,22,25],string:[3,5,11,16,18,20,21],stub:11,subclass:[1,2,3,4,5,7,10,11,12,13,14,17,18,20,22,25],sublist:19,submit:11,subsequ:13,subset:21,subtract:1,subtyp:[1,22,25],suffix:[1,3,4,7,10,11,13,14,18,20,22,25],sum:[1,7,10,11,13,25],summari:[1,25],support:[19,21],suppress:[5,20],synonym:16,system:[2,5],system_bas:21,systemcontrol:11,systemdefaultccontrol:2,systemdefaultinputmechan:11,systemdefaultoutputmechan:11,systemdefaultreceiv:10,systemdefaultsend:10,systemregistri:21,take:[3,11,16,18,20,21],take_over_as_default_control:2,target:[1,11,16],target_input:19,target_set:[2,11,25],task:[3,4],tbi:[2,3,4,11,18,21],tc_predic:1,tempor:[3,4,22],term:[4,19],termin:[1,2,4,11,16,19,21,22],terminal_mechan:21,terminalmechan:[16,21],terminate_execut:11,terminate_funct:[1,4,22],test:[1,12,16,18,20],than:[1,3,4,10,11,13,14,16,18,19,20,22,25],thei:[1,4,7,11,14,16,18,21,22,25],them:[2,3,7,10,13,14,18,20],themselv:16,theoret:21,therefor:16,thi:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],third:16,those:[1,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],three:[3,11,16,18,20,21],threshold:4,through:[11,16,18,21],throughout:16,thu:11,time:[4,7,10,11,13,14,16],time_scal:[1,2,3,4,10,11,14,16,19,20,21,22,25],time_step:[4,11,16,19,21],time_step_s:4,timescal:[1,2,4,11,14,16,19,20,21,22,25],togeth:[11,18,21],tool:21,topolog:21,toposort:21,total:[3,4],total_alloc:4,total_cost:4,total_cost_funct:3,totalcost:3,totalcostfunct:3,track:5,train:[11,16],trajectori:4,transfer:[11,17],transfer_default_bia:22,transfer_default_rang:22,transfer_preferenceset:22,transfer_rang:22,transferouput:22,transform:[2,10,11,16,21,22],translat:3,transmit:16,treatment:19,trial:[1,2,4,11,14,16,19,20,21,22,25],tupl:[2,3,5,11,14,16,20,21],tuples_list:[11,16],turn:20,two:[1,4,7,11,13,14,16,18,19,20,21,25],type:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],typecheck:[1,16,19,21],typic:[11,19],unchang:[5,10],under:[10,11,16,20,21],understand:19,uniqu:3,unit:22,unless:[1,4,14,22,25],until:[1,4,11,22,25],updat:[2,3,4,7,10,11,13,14,19,20],update_and_execut:11,update_control_sign:3,update_monitored_state_changed_attribut:12,update_st:[7,14,20],update_states_and_execut:[11,18],upon:19,user:20,usual:[11,19],util:[1,3,4,7,10,11,13,14,22],valid:[3,7,10,11,13,14,18,19,20,25],validate_monitoredstates_spec:2,valu:[1,2,3,4,5,7,10,11,12,13,14,16,18],variabilti:4,variabl:[1,2,3,4,7,10,11,12,13,14,16,17,18,19,20,22,25],variable_default:2,variableclassdefault:[1,4,11,18,20,22,25],variableindex:[3,5],variableinstancedefault:16,variablevalu:[3,5],varianc:[1,22,25],vector:[1,3,10,19],version:4,wai:[1,3,4,7,10,11,13,14,16,18,20,21,22,25],warn:20,weight:[2,10,11,21],weightederror:25,weightederror_preferenceset:25,well:[1,4,25],what:[1,5],when:[2,7,11,13,16,19,21],whenev:11,where:[11,16,20,21],whether:[1,2,11,12,16,18,19,20,21],which:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],whose:21,why:3,width:25,wiener:4,wise:[1,4],within:[7,10,11,13,16,18,19,20,21],without:[11,18,20],would:[11,19],xxx:[14,16,20],zero:[16,21]},titles:["Adaptive Integrator","Comparator","Control Mechanisms","Control Signal","DDM","Default Control Mechanism","&lt;no title&gt;","Input State","Learning","Log","Mapping","Mechanisms","Monitoring Mechanism","Output State","Parameter State","Preferences","Process","Processing Mechanisms","Projections","Run","States","System","Transfer","Utilities","Utility Function","Weighted Error","Welcome to PsyNeuLink&#8217;s documentation!"],titleterms:{"default":5,"function":[11,24],adapt:0,compar:1,control:[2,3,5,21],custom:11,ddm:4,document:26,entri:9,error:25,execut:21,indic:26,initi:[19,21],input:[7,16,19,21],integr:0,learn:[8,16,21],log:9,map:10,mechan:[2,5,11,12,16,17],monitor:12,order:21,output:[13,16],overview:[11,16,19,21],paramet:14,phase:21,prefer:15,process:[11,16,17],project:[16,18],psyneulink:26,role:11,run:19,signal:3,state:[7,11,13,14,20],structur:[16,21],system:[11,21],tabl:26,target:19,time:19,transfer:22,util:[23,24],valu:19,weight:25,welcom:26}})