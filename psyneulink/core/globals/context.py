# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ********************************************  System Defaults ********************************************************

"""
.. _Context_Overview:

Overview
--------
The Context class is used for the `context <Component.context>` attribute of all `Components <Component>`.  It is
set when a Component is first instantiated, and updated under various operating conditions.  Its primary
attribute is `flags <Context.flags>` - a binary vector, the individual flags of which are specified using the
`ContextFlags` enum.  The `flags <Context.flags>` attribute is divided functionally into the two fields:

  * `execution_phase <Context.execution_phase>` - phase of execution of the Component;

  * `source <Context.source>` - source of a call to a method belonging to or operating on the Component.

Each field can be addressed using the corresponding property of the class, and in general only one of the flags
in a field is set (although see individual property documentation for exceptions).

Context and Logging
-------------------

The `flags <Context.flags>` attribute is used by `Log` to identify conditions for logging (see).  Accordingly, the
`LogCondition`\\(s) used to specify such conditions in the `set_log_conditions <Log.set_log_conditions>` method of Log
are a subset of (and are aliased to) the flags in `ContextFlags`.

.. _Context_Additional_Attributes:

Additional Attributes
---------------------

In addition to `flags <Context.flags>`, Context has four other attributes that record information relevant to the
operating state of the Component:

    `owner <Context.owner>`
      the Component to which the Context belongs (assigned to its `context <Component.context>` attribute;
    `flags_string <Context.flags_string>`
      a string containing the names of the flags currently set in each of the fields of the `flags <Context.flags>`
      attribute;
    `composition <Context.composition>`
      the `Composition <Composition>` in which the Component is currently being executed;
    `execution_id <Context.execution_id>`
      the `execution_id` assigned to the Component by the Composition in which it is currently being executed;
    `execution_time <Context.execution_time>`
      the current time of the scheduler running the Composition within which the Component is currently being executed;
    `string <Context.string>`
      contains message(s) relevant to a method of the Component currently invoked or that is referencing the Component.
      In general, this contains a copy of the **context** argument passed to method of the Component or one that
      references it, but it is possible that future uses will involve other messages.

    .. _Context_String_Note:

    .. note::
       The `string <Context.string>` attribute of Context is not the same as, nor does it usually contain the same
       information as the string returned by the `flags_string <Context.flags_string>` method of Context.

COMMENT:
    IMPLEMENTATION NOTE: Use of ContextFlags in **context** argument of methods for context message-passing
        ContextFlags is also used for passing context messages to methods (in the **context** argument).

        Among other things, this is used to determine the source of call of a constructor (until someone
            proposes/implements a better method!).  This is used in several ways, for example:
            a) to insure that any call to a _Base class is from a subclass constructor
              rather than by the user from the command line (which is not allowed).
            b) to determine whether an InputState or OutputState is being added as part of the construction process
              (e.g., for LearningMechanism) or by the user from the command line (see Mechanism.add_states)

        Application (a) above is implemented as follows:
            * user-accessible subclasses do not implement a context argument in their constructors
            * subclasses just below the _Base class call super().__init__() with context=ContextFlags.CONSTRUCTOR
            * all lower sub-subclasses call super without context arg
            * the constructor for all _Base classes checks that context==ContextFlags.CONSTRUCTOR
              and assign self.context.source = context
            * the constructor for all _Base classes do NOT pass a context arg to super (i.e., shellclasses or Component)
              since Component.__init__() assigns context arg as CONSTRUCTOR for all of its calls
COMMENT

.. _Context_Class_Reference:

Class Reference
---------------

"""

import enum
import warnings

from collections import namedtuple
from uuid import UUID

import typecheck as tc

from psyneulink.core.globals.keywords import CONTROL, EXECUTING, EXECUTION_PHASE, FLAGS, INITIALIZATION_STATUS, INITIALIZING, LEARNING, SEPARATOR_BAR, SOURCE, VALIDATE
from psyneulink.core.globals.utilities import get_deepcopy_with_shared


__all__ = [
    'Context',
    'ContextFlags',
    '_get_context',
    'INITIALIZATION_STATUS_FLAGS',
]

STATUS = 'status'

time = namedtuple('time', 'run trial pass_ time_step')

class ContextError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ContextFlags(enum.IntFlag):
    """Used to identify the initialization and execution status of a `Component <Component>`.

    Used when a Component's `value <Component.value>` or one of its attributes is being accessed.
    Also used to specify the context in which a value of the Component or its attribute is `logged <Log_Conditions>`..

    COMMENT:
        Used to by **context** argument of all methods to specify type of caller.
    COMMENT
    """

    UNSET = 0

    DEFERRED_INIT = 1<<1  # 2
    """Set if flagged for deferred initialization."""
    INITIALIZING =  1<<2  # 4
    """Set during initialization of the Component."""
    VALIDATING =    1<<3  # 8
    """Set during validation of the value of a Component or its attribute."""
    INITIALIZED =   1<<4  # 16
    """Set after completion of initialization of the Component."""
    REINITIALIZED =   1<<4  # 16
    """Set on stateful Components when they are re-initialized."""
    UNINITIALIZED = 1 << 16
    """Default value set before initialization"""

    INITIALIZATION_MASK = DEFERRED_INIT | INITIALIZING | VALIDATING | INITIALIZED | REINITIALIZED | UNINITIALIZED

    # execution_phase flags
    PROCESSING =    1<<5  # 32
    """Set during the `processing phase <System_Execution_Processing>` of execution of a Composition."""
    LEARNING =      1<<6 # 64
    """Set during the `learning phase <System_Execution_Learning>` of execution of a Composition."""
    CONTROL =       1<<7 # 128
    """Set during the `control phase System_Execution_Control>` of execution of a Composition."""
    SIMULATION =    1<<8  # 256
    """Set during simulation by Composition.controller"""
    IDLE = 1 << 17
    """Identifies condition in which no flags in the `execution_phase <Context.execution_phase>` are set.
    """

    EXECUTING = PROCESSING | LEARNING | CONTROL | SIMULATION
    EXECUTION_PHASE_MASK = EXECUTING | IDLE

    # source (source-of-call) flags
    COMMAND_LINE =  1<<9  # 512
    """Direct call by user (either interactively from the command line, or in a script)."""
    CONSTRUCTOR =   1<<10 # 1024
    """Call from Component's constructor method."""
    INSTANTIATE =   1<<11 # 2048
    """Call by an instantiation method."""
    COMPONENT =     1<<12 # 4096
    """Call by Component __init__."""
    METHOD =        1<<13 # 8192
    """Call by method of the Component other than its constructor."""
    PROPERTY =      1<<14 # 16384
    """Call by property of the Component."""
    COMPOSITION =   1<<15 # 32768
    """Call by a/the Composition to which the Component belongs."""

    PROCESS =   1<<15     # 32768
    NONE = 1 << 18

    """Call by a/the Composition to which the Component belongs."""
    SOURCE_MASK = COMMAND_LINE | CONSTRUCTOR | INSTANTIATE | COMPONENT | METHOD | PROPERTY | COMPOSITION | PROCESS | NONE


    ALL_FLAGS = INITIALIZATION_MASK | EXECUTION_PHASE_MASK | SOURCE_MASK

    @classmethod
    @tc.typecheck
    def _get_context_string(cls, condition_flags,
                            fields:tc.any(tc.enum(EXECUTION_PHASE,
                                                  SOURCE), set, list)={EXECUTION_PHASE,
                                                                       SOURCE},
                            string:tc.optional(str)=None):
        """Return string with the names of flags that are set in **condition_flags**

        If **fields** is specified, then only the names of the flag(s) in the specified field(s) are returned.
        The fields argument must be the name of a field (*EXECUTION_PHASE* or *SOURCE*)
        or a set or list of them.

        If **string** is specified, the string returned is prepended by **string**.
        """

        if string:
            string += ": "
        else:
            string = ""

        if isinstance(fields, str):
            fields = {fields}

        flagged_items = []
        # If OFF or ALL_FLAGS, just return that
        if condition_flags == ContextFlags.ALL_FLAGS:
            return ContextFlags.ALL_FLAGS.name
        if condition_flags == ContextFlags.UNSET:
            return ContextFlags.UNSET.name
        # Otherwise, append each flag's name to the string
        # for c in (EXECUTION_PHASE_FLAGS | SOURCE_FLAGS):
        #     if c & condition_flags:
        #        flagged_items.append(c.name)
        if EXECUTION_PHASE in fields:
            for c in EXECUTION_PHASE_FLAGS:
                if not condition_flags & ContextFlags.EXECUTION_PHASE_MASK:
                    flagged_items.append(ContextFlags.IDLE.name)
                    break
                if c & condition_flags:
                   flagged_items.append(c.name)
        if SOURCE in fields:
            for c in SOURCE_FLAGS:
                if not condition_flags & ContextFlags.SOURCE_MASK:
                    flagged_items.append(ContextFlags.NONE.name)
                    break
                if c & condition_flags:
                   flagged_items.append(c.name)
        string += ", ".join(flagged_items)
        return string

INITIALIZATION_STATUS_FLAGS = {ContextFlags.DEFERRED_INIT,
                               ContextFlags.INITIALIZING,
                               ContextFlags.VALIDATING,
                               ContextFlags.INITIALIZED,
                               ContextFlags.REINITIALIZED,
                               ContextFlags.UNINITIALIZED}

EXECUTION_PHASE_FLAGS = {ContextFlags.PROCESSING,
                         ContextFlags.LEARNING,
                         ContextFlags.CONTROL,
                         ContextFlags.SIMULATION,
                         ContextFlags.IDLE
                         }

SOURCE_FLAGS = {ContextFlags.COMMAND_LINE,
                ContextFlags.CONSTRUCTOR,
                ContextFlags.INSTANTIATE,
                ContextFlags.COMPONENT,
                ContextFlags.METHOD,
                ContextFlags.PROPERTY,
                ContextFlags.COMPOSITION,
                ContextFlags.NONE}


class Context():
    """Used to indicate the state of initialization and phase of execution of a Component, as well as the source of
    call of a method;  also used to specify and identify `conditions <Log_Conditions>` for `logging <Log>`.


    Attributes
    ----------

    owner : Component
        Component to which the Context belongs.

    flags : binary vector
        represents the current operating context of the `owner <Context.owner>`; contains two fields
        `execution_phase <Context.execution_phase>`,
        and `source <Context.source>` (described below).

    flags_string : str
        contains the names of the flags currently set in each of the fields of the `flags <Context.flags>` attribute;
        note that this is *not* the same as the `string <Context.string>` attribute (see `note <Context_String_Note>`).

    execution_phase : field of flags attribute
        indicates the phase of execution of the Component;
        one or more of the following flags can be set:

            * `PROCESSING <ContextFlags.PROCESSING>`
            * `LEARNING <ContextFlags.LEARNING>`
            * `CONTROL <ContextFlags.CONTROL>`
            * `SIMULATION <ContextFlags.SIMULATION>`
            * `IDLE <ContextFlags.IDLE>`

        If `IDLE` is set, the Component is not being executed at the current time, and `flags_string
        <Context.flags_string>` will include *IDLE* in the string.  In some circumstances all of the
        `execution_phase <Context.execution_phase>` flags may be set, in which case `flags_string
        <Context.flags_string>` will include *EXECUTING* in the string.

    source : field of the flags attribute
        indicates the source of a call to a method belonging to or referencing the Component;
        one of the following flags is always set:

            * `CONSTRUCTOR <ContextFlags.CONSTRUCTOR>`
            * `COMMAND_LINE <ContextFlags.COMMAND_LINE>`
            * `COMPONENT <ContextFlags.COMPONENT>`
            * `COMPOSITION <ContextFlags.COMPOSITION>`

    composition : Composition
      the `Composition <Composition>` in which the `owner <Context.owner>` is currently being executed.

    execution_id : UUID
      the execution_id assigned to the Component by the Composition in which it is currently being executed.

    execution_time : TimeScale
      current time of the `Scheduler` running the Composition within which the Component is currently being executed.

    string : str
      contains message(s) relevant to a method of the Component currently invoked or that is referencing the Component.
      In general, this contains a copy of the **context** argument passed to method of the Component or one that
      references it, but it is possible that future uses will involve other messages.  Note that this is *not* the
      same as the `flags_string <Context.flags_string>` attribute (see `note <Context_String_Note>`).

    """

    __name__ = 'Context'
    _deepcopy_shared_keys = {'owner', 'composition', '_composition'}

    def __init__(self,
                 owner=None,
                 composition=None,
                 flags=None,
                 execution_phase=ContextFlags.IDLE,
                 # source=ContextFlags.COMPONENT,
                 source=ContextFlags.NONE,
                 execution_id:UUID=None,
                 string:str='', time=None):

        self.owner = owner
        self.composition = composition
        self._execution_phase = execution_phase
        self._source = source
        if flags:
            if (execution_phase and not (flags & ContextFlags.EXECUTION_PHASE_MASK & execution_phase)):
                raise ContextError("Conflict in assignment to flags ({}) and execution_phase ({}) arguments "
                                   "of Context for {}".
                                   format(ContextFlags._get_context_string(flags & ContextFlags.EXECUTION_PHASE_MASK),
                                          ContextFlags._get_context_string(flags, EXECUTION_PHASE), self.owner.name))
            if (source != ContextFlags.COMPONENT) and not (flags & ContextFlags.SOURCE_MASK & source):
                raise ContextError("Conflict in assignment to flags ({}) and source ({}) arguments of Context for {}".
                                   format(ContextFlags._get_context_string(flags & ContextFlags.SOURCE_MASK),
                                          ContextFlags._get_context_string(flags, SOURCE),
                                          self.owner.name))
        self.execution_id = execution_id
        self.execution_time = None
        self.string = string

    __deepcopy__ = get_deepcopy_with_shared(_deepcopy_shared_keys)

    @property
    def composition(self):
        try:
            return self._composition
        except AttributeError:
            self._composition = None

    @composition.setter
    def composition(self, composition):
        # from psyneulink.core.compositions.composition import Composition
        # if isinstance(composition, Composition):
        if (
            composition is None
            or composition.__class__.__name__ in {
                'Composition', 'SystemComposition', 'PathwayComposition', 'AutodiffComposition', 'System', 'Process'
            }
        ):
            self._composition = composition
        else:
            raise ContextError("Assignment to context.composition for {} ({}) "
                               "must be a Composition (or \'None\').".format(self.owner.name, composition))

    @property
    def flags(self):
        return self.execution_phase | self.source

    @flags.setter
    def flags(self, flags: ContextFlags):
        if isinstance(flags, (ContextFlags, int)):
            self.execution_phase = flags & ContextFlags.EXECUTION_PHASE_MASK
            self.source = flags & ContextFlags.SOURCE_MASK
        else:
            raise ContextError("\'{}\'{} argument in call to {} must be a {} or an int".
                               format(FLAGS, flags, self.__name__, ContextFlags.__name__))

    @property
    def flags_string(self):
        return ContextFlags._get_context_string(self.flags)

    @property
    def execution_phase(self):
        return self._execution_phase

    @execution_phase.setter
    def execution_phase(self, flag):
        """Check that flag is a valid execution_phase flag assignment"""
        if not flag:
            self._execution_phase = ContextFlags.IDLE
        elif (flag & ~ContextFlags.EXECUTION_PHASE_MASK):
            raise ContextError("Attempt to assign a flag ({}) to execution_phase "
                               "that is not an execution phase flag".
                               format(str(flag)))
        else:
            if (
                flag in EXECUTION_PHASE_FLAGS
                or (flag & ~(ContextFlags.SIMULATION | ContextFlags.CONTROL)) in EXECUTION_PHASE_FLAGS
            ):
                self._execution_phase = flag
            else:
                raise ContextError(
                    f"Attempt to assign more than one non-SIMULATION/CONTROL flag ({str(flag)}) to execution_phase"
                )

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, flag):
        """Check that a flag is one and only one source flag"""
        if flag in SOURCE_FLAGS:
            self._source = flag
        elif not flag:
            self._source = ContextFlags.NONE
        elif not flag & ContextFlags.SOURCE_MASK:
            raise ContextError("Attempt to assign a flag ({}) to source that is not a source flag".
                               format(str(flag)))
        else:
            raise ContextError("Attempt to assign more than one flag ({}) to source".
                               format(str(flag)))

    @property
    def execution_time(self):
        try:
            return self._execution_time
        except AttributeError:
            return None

    @execution_time.setter
    def execution_time(self, time):
        self._execution_time = time

    def update_execution_time(self):
        if self.execution & ContextFlags.EXECUTING:
            self.execution_time = _get_time(self.owner, self.context.flags)
        else:
            raise ContextError("PROGRAM ERROR: attempt to call update_execution_time for {} "
                               "when 'EXECUTING' was not in its context".format(self.owner.name))

    def add_to_string(self, string):
        if self.string is None:
            self.string = string
        else:
            self.string = '{0} {1} {2}'.format(self.string, SEPARATOR_BAR, string)

    def _change_flags(self, *flags, operation=lambda attr, blank_flag, *flags: NotImplemented):
        # split by flag type to avoid extra costly binary operations on enum flags
        if all([flag in EXECUTION_PHASE_FLAGS for flag in flags]):
            self.execution_phase = operation(self.execution_phase, ContextFlags.IDLE, *flags)
        elif all([flag in SOURCE_FLAGS for flag in flags]):
            self.source = operation(self.source, ContextFlags.NONE, *flags)
        else:
            raise ContextError(f'Flags must all correspond to one of: execution_phase, source')

    def add_flag(self, flag: ContextFlags):
        def add(attr, blank_flag, flag):
            return (attr & ~blank_flag) | flag

        self._change_flags(flag, operation=add)

    def remove_flag(self, flag: ContextFlags):
        def remove(attr, blank_flag, flag):
            if attr & flag:
                res = (attr | flag) ^ flag
                if res is ContextFlags.UNSET:
                    res = blank_flag
                return res
            else:
                return attr

        self._change_flags(flag, operation=remove)

    def replace_flag(self, old: ContextFlags, new: ContextFlags):
        def replace(attr, blank_flag, old, new):
            return (attr & ~old) | new

        self._change_flags(old, new, operation=replace)


@tc.typecheck
def _get_context(context:tc.any(ContextFlags, str)):
    """Set flags based on a string of ContextFlags keywords
    If context is already a ContextFlags mask, return that
    Otherwise, return mask with flags set corresponding to keywords in context
    """
    # FIX: 3/23/18 UPDATE WITH NEW FLAGS
    if isinstance(context, ContextFlags):
        return context
    context_flag = ContextFlags.UNSET
    if VALIDATE in context:
        context_flag |= ContextFlags.VALIDATING
    if EXECUTING in context:
        context_flag |= ContextFlags.EXECUTING
    if CONTROL in context:
        context_flag |= ContextFlags.CONTROL
    if LEARNING in context:
        context_flag |= ContextFlags.LEARNING
    # if context == ContextFlags.TRIAL.name: # cxt-test
    #     context_flag |= ContextFlags.TRIAL
    # if context == ContextFlags.RUN.name:
    #     context_flag |= ContextFlags.RUN
    if context == ContextFlags.COMMAND_LINE.name:
        context_flag |= ContextFlags.COMMAND_LINE
    return context_flag


def _get_time(component, context_flags, execution_id=None):
    """Get time from Scheduler of System in which Component is being executed.

    Returns tuple with (run, trial, time_step) if being executed during Processing or Learning
    Otherwise, returns (None, None, None)

    """

    from psyneulink.core.globals.context import time
    from psyneulink.core.components.mechanisms.mechanism import Mechanism
    from psyneulink.core.components.states.state import State
    from psyneulink.core.components.projections.projection import Projection

    no_time = time(None, None, None, None)

    # Get mechanism to which Component being logged belongs
    if isinstance(component, Mechanism):
        ref_mech = component
    elif isinstance(component, State):
        if isinstance(component.owner, Mechanism):
            ref_mech = component.owner
        elif isinstance(component.owner, Projection):
            ref_mech = component.owner.receiver.owner
        else:
            raise ContextError("Logging currently does not support {} (only {}s, {}s, and {}s).".
                           format(component.__class__.__name__,
                                  Mechanism.__name__, State.__name__, Projection.__name__))
    elif isinstance(component, Projection):
        ref_mech = component.receiver.owner
    else:
        raise ContextError("Logging currently does not support {} (only {}s, {}s, and {}s).".
                       format(component.__class__.__name__,
                              Mechanism.__name__, State.__name__, Projection.__name__))

    # FIX: Modify to use component.owner.context.composition once that is implemented
    # Get System in which it is being (or was last) executed (if any):

    # If called from COMMAND_LINE, get context for last time value was assigned:
    if context_flags & ContextFlags.COMMAND_LINE:
    # if context_flags & (ContextFlags.COMMAND_LINE | ContextFlags.RUN | ContextFlags.TRIAL):
        if component.prev_context:
            context_flags = component.prev_context.flags
        else:
            context_flags = ContextFlags.UNSET

    system = ref_mech.parameters.context._get(execution_id).composition

    if system and hasattr(system, 'scheduler_processing'):
        execution_flags = context_flags & ContextFlags.EXECUTION_PHASE_MASK
        # # MODIFIED 7/15/19 OLD:
        # try:
        #     if execution_flags == ContextFlags.PROCESSING or not execution_flags:
        #         t = system.scheduler_processing.clocks[execution_id].time
        #         t = time(t.run, t.trial, t.pass_, t.time_step)
        #     elif execution_flags == ContextFlags.CONTROL:
        #         t = system.scheduler_processing.clocks[execution_id].time
        #         t = time(t.run, t.trial, t.pass_, t.time_step)
        #     elif execution_flags == ContextFlags.LEARNING:
        #         if hasattr(system, "scheduler_learning") and system.scheduler_learning is not None:
        #             t = system.scheduler_learning.clocks[execution_id].time
        #             t = time(t.run, t.trial, t.pass_, t.time_step)
        #         # KAM HACK 2/13/19 to get hebbian learning working for PSY/NEU 330
        #         # Add autoassociative learning mechanism + related projections to composition as processing components
        #         else:
        #             t = None
        #     else:
        #         t = None
        # MODIFIED 7/15/19 NEW:  ACCOMODATE LEARNING IN COMPOSITION DONE WITH scheduler_processing
        try:
            if execution_flags in {ContextFlags.PROCESSING, ContextFlags.LEARNING, ContextFlags.IDLE}:
                t = system.scheduler_processing.clocks[execution_id].time
                t = time(t.run, t.trial, t.pass_, t.time_step)
            elif execution_flags == ContextFlags.CONTROL:
                t = system.scheduler_processing.clocks[execution_id].time
                t = time(t.run, t.trial, t.pass_, t.time_step)
            # elif execution_flags == ContextFlags.LEARNING:
            #     if hasattr(system, "scheduler_learning") and system.scheduler_learning is not None:
            #         t = system.scheduler_learning.clocks[execution_id].time
            #         t = time(t.run, t.trial, t.pass_, t.time_step)
            #     # KAM HACK 2/13/19 to get hebbian learning working for PSY/NEU 330
            #     # Add autoassociative learning mechanism + related projections to composition as processing components
            #     else:
            #         t = None
            else:
                t = None
        # MODIFIED 7/15/19 END:
        except KeyError:
            t = None

    else:
        if component.verbosePref:
            offender = "\'{}\'".format(component.name)
            if ref_mech is not component:
                offender += " [{} of {}]".format(component.__class__.__name__, ref_mech.name)
            warnings.warn("Attempt to log {} which is not in a System (logging is currently supported only "
                          "when running Components within a System".format(offender))
        t = None

    return t or no_time
