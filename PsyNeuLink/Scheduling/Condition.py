# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Condition **************************************************************

"""

.. _Condition_Overview

Overview
--------

`Conditions <Condition>` are used to specify when `Mechanisms <Mechanism>` are allowed to execute.  Conditions
can be used to specify a variety of required conditions for execution, including the state of the Mechanism
itself (e.g., how many times it has already executed, or the value of one of its attributes), the state of the
Composition (e.g., how many `TIME_STEPs <TIME_STEP>` have occurred in the current `TRIAL`), or the state of other
Mechanisms in a Composition (e.g., whether they have started, terminated, or how many times they have executed).
PsyNeuLink provides a number of `pre-specified Conditions <Condition_Structure>` that can be parametrized
(e.g., how many times a Mechanism should be executed), but functions can also be assigned to Conditions,
to implement custom conditions that can reference any object or its attributes in PsyNeuLink.

.. _Condition_Instantiation:

Instantiating Conditions
------------------------

Conditions can be instantiated and added to a `Scheduler` at any time, and take effect immediately for the
execution of that Scheduler. Each `pre-specified Condition <Condition_Structure>` has a set of arguments
that must be specified to achieve the desired behavior. Most
Conditions are associated with an `owner <Condition.owner>` (a `Mechanism` to which the Condition belongs), and a
`scheduler <Condition.scheduler>` that maintains most of the data required to test for satisfaction of the condition.
Usually, Conditions may be instantiated within a call to `Scheduler` or `ConditionSet`'s add methods, in which case
the attributes `owner` and `scheduler` are determined through context, as below::

    scheduler.add_condition(A, EveryNPasses(1))
    scheduler.add_condition(B, EveryNCalls(A, 2))
    scheduler.add_condition(C, EveryNCalls(B, 2))


.. _Condition_Creation:

Custom Conditions
-----------------------

Arbitrary `Conditions <Condition>` may be created and used *ad hoc*, using the base `Condition` class.
COMMENT:
    K: Thinking about it I kind of like making basic wrappers While and Until, where While is exactly the same as
        base Condition, but perhaps more friendly sounding? It evals to the output of the function exactly
        Until would just be the inversion of the function. Thoughts?
    JDC: THIS SOUNDS GOOD.
COMMENT

In order to construct a custom Condition object, a function (or other callable) must be passed to the Condition's
**func** argument. This is used to specify the function that will be evaluated on each `PASS` through the Mechanisms
in the Composition, to determine whether the associated Mechanism is allowed to execute on that `PASS`. After the
function, additional args and kwargs may be passed to the constructor; the function will be called with these
parameters upon evaluation of the Condition. Custom Conditions can provide for expressive behavior,
such as satisfaction after a recurrent mechanism has converged::

    def converge(mech, thresh):
        return abs(mech.value - mech.previous_value) < thresh

    Until(
        converge,
        my_recurrent_mech,
        epsilon
    )



.. _Condition_Structure:

Structure
---------

The Scheduler associates every Mechanism with a Condition.  If a Condition has not been explicitly specified for a
Mechanism, the Mechanism is assigned the Condition `Always`, which allows it to be executed whenever it is
`under consideration <Scheduler_Algorithm>`.  As described `above <Condition_Creation>`, there are pre-specified
subclasses (listed below) that implement standard conditions and simply require the specification of a parameter
or two to be implemented, while the generic Condition can be used to implement custom conditions by passing it a
function and associated arguments.  Following is a list of pre-specified Conditions and the parameters they require
(described in greater detail under `Condition_Class_Reference`):

.. note::
    The optional `TimeScale` argument in many of the `Conditions <Condition>` below specifies the scope over which the
    Condition operates;  this defaults to `TRIAL` in all cases, except for Conditions with "Trial" in their name
    in which it defaults to `RUN`.

.. _Conditions_Static:

COMMENT:
    K: I don't think we need to comment on how Always causes execution in its description,
    because it's mentioned right above
COMMENT

**Static Conditions** (independent of other Conditions, Components or time):

    * `Always`
      \
      always satisfied.

    * `Never`
      \
      never satisfied.


.. _Conditions_Composite:

**Composite Conditions** (based on other Conditions):

    * `All`\ (Conditions)
      \
      satisfied whenever all of the specified Conditions are satisfied.

    * `Any`\ (Conditions)
      \
      satisfied whenever any of the specified Conditions are satisfied.

    * `Not`\ (Condition)
      \
      satisfied whenever the specified Condition is not satisfied.


.. _Conditions_Time_Based:

**Time-Based Conditions** (based on time at a specified `TimeScale`):


    * `BeforePass`\ (int[, TimeScale])
      \
      satisfied any time before the specified `PASS` occurs.

    * `AtPass`\ (int[, TimeScale])
      \
      satisfied only during the specified `PASS`.

    * `AfterPass`\ (int[, TimeScale])
      \
      satisfied any time after the specified `PASS` has occurred.

    * `AfterNPasses`\ (int[, TimeScale])
      \
      satisfied when or any time after the specified number of `PASS`\es has occurred.

    * `EveryNPasses`\ (int[, TimeScale])
      \
      satisfied every time the specified number of `PASS`\ es occurs.

    * `BeforeTrial`\ (int[, TimeScale])
      \
      satisfied any time before the specified `TRIAL` occurs.

    * `AtTrial`\ (int[, TimeScale])
      \
      satisfied any time during the specified `TRIAL`.

    * `AfterTrial`\ (int[, TimeScale])
      \
      satisfied any time after the specified `TRIAL` occurs.

    * `AfterNTrials`\ (int[, TimeScale])
      \
      satisfied any time after the specified number of `TRIAL`\s has occurred.


.. _Conditions_Component_Based:

**Component-Based Conditions** (based on the execution or state of other Components):


    * `BeforeNCalls`\ (Component, int[, TimeScale])
      \
      satisfied any time before the specified Component has executed the specified number of times.

    * `AtNCalls`\ (Component, int[, TimeScale])
      \
      satisfied when the specified Component has executed the specified number of times.

    * `AfterCall`\ (Component, int[, TimeScale])
      \
      satisfied any time after the Component has executed the specified number of times.

    * `AfterNCalls`\ (Component, int[, TimeScale])
      \
      satisfied when or any time after the Component has executed the specified number of times.

    * `AfterNCallsCombined`\ (Components, int[, TimeScale])
      \
      satisfied when or any time after the specified Components have executed the specified number
      of times among themselves, in total.

    * `EveryNCalls`\ (Component, int[, TimeScale])
      \
      satisfied when the specified Component has executed the specified number of times since the
      last time `owner` has run.

    * `JustRan`\ (Component)
      \
      satisfied if the specified Component was assigned to run in the previous `TIME_STEP`.

    * `AllHaveRun`\ (Components)
      \
      satisfied when all of the specified Components have executed at least once.

    * `WhenFinished`\ (Component)
      \
      satisfied when the specified Component has finished (i.e, its `is_finished` attribute is `True`).

    * `WhenFinishedAny`\ (Components)
      \
      satisfied when any of the specified Components have finished (i.e, `is_finished` is `True` for any of them).

    * `WhenFinishedAll`\ (Components)
      \
      satisfied when all of the specified Components have finished (i.e, `is_finished` is `True` for all of them).


.. Condition_Execution:

Execution
---------

When the `Scheduler` `runs <Schedule_Execution>`, it makes a sequential `PASS` through its `consideration_queue`,
evaluating each `consideration_set` in the queue to determine which Mechanisms should be assigned to execute.
It evaluates the Mechanisms in each set by calling the `is_satisfied` method of the Condition associated with each
of those Mechanisms.  If it returns `True`, then the Mechanism is assigned to the execution set for the `TIME_STEP`
of execution generated by that `PASS`.  Otherwise, the Mechanism is not executed.

.. _Condition_Class_Reference:

Class Reference
---------------

"""

import logging

from PsyNeuLink.Globals.TimeScale import TimeScale

logger = logging.getLogger(__name__)


class ConditionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ConditionSet(object):
    """Used in conjunction with a `Scheduler` to store the `Conditions <Condition>` associated with a `Component`.

    Arguments
    ---------

    scheduler : Scheduler
        specifies the `Scheduler` used to evaluate and maintain a record of the information required to
        evalute the `Conditions <Condition>`

    conditions : dict{Component: Condition}
        specifies an iterable collection of `Components <Component>` and the `Conditions <Condition>` associated
        with each.

    Attributes
    ----------

    scheduler : Scheduler
        specifies the `Scheduler` used to evaluate and maintain a record of the information required to
        evalute the `Conditions <Condition>`

    COMMENT:
        JDC: IN THE ATTRIBUTE IS IT CONVERTED TO A STANDARD ITERABLE FORM (E.G. A LIST)?  IF SO, SHOULD
        MODIFY DESCRIPTION BELOW ACCORDINGLY
    COMMENT
    conditions : dict{Component: Condition}
        the key of each entry is a `Component`, and its value is the `Condition <Condition>` associated
        with that Component.  Conditions can be added to the
        ConditionSet using the ConditionSet's `add_condition` method.

    """
    def __init__(self, scheduler=None, conditions=None):
        self.conditions = conditions if conditions is not None else {}
        self.scheduler = scheduler

    def __contains__(self, item):
        return item in self.conditions

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        logger.debug('ConditionSet ({0}) setting scheduler to {1}'.format(type(self).__name__, value))
        self._scheduler = value

        for owner, cond in self.conditions.items():
            cond.scheduler = value

    def add_condition(self, owner, condition):
        """Add a `Condition` to the ConditionSet.

        Arguments
        ---------

        owner : Component
            specifies the Component with which the **condition** should be associated.

        condition : Condition
            specifies the Condition, associated with the **owner** to be added to the ConditionSet.


        """
        logger.debug('add_condition: Setting scheduler of {0}, (owner {2}) to self.scheduler ({1})'.
                     format(condition, self.scheduler, owner))
        condition.owner = owner
        condition.scheduler = self.scheduler
        self.conditions[owner] = condition

    def add_condition_set(self, conditions):
        """Add a collection of `Conditions <Condition>` to the ConditionSet.

        Arguments
        ---------

        conditions : dict{Component: Condition}
            specifies an iterable collection of Conditions to be added to the ConditionSet, in the form of a dict
            each entry of which maps a `Component` (the key) to a `Condition <Condition>` (the value).

        """
        for owner in conditions:
            conditions[owner].owner = owner
            conditions[owner].scheduler = self.scheduler
            self.conditions[owner] = conditions[owner]


class Condition(object):
    """
    Used in conjunction with a `Scheduler` to specify the condition under which a `Mechanism` should be
    allowed to execute.

    Arguments
    ---------

    func : callable
        specifies function to be called when the Condition is evaluated, to determine whether it is currently satisfied.

    args : *args
        specifies formal arguments to pass to `func` when the Condition is evaluated.

    kwargs : **kwargs
        specifies keyword arguments to pass to `func` when the Condition is evaluated.

    Attributes
    ----------

    scheduler : Scheduler
        the `Scheduler` with which the Condition is associated;  the Scheduler's state is used to evaluate whether
        the Condition`s specifications are satisfied.

    owner (Component):
        the `Component` with which the Condition is associated, and the execution of which it determines.

        """
    def __init__(self, dependencies, func, *args, **kwargs):
        self.dependencies = dependencies
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self._scheduler = None
        self._owner = None

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        logger.debug('Condition ({0}) setting scheduler to {1}'.format(type(self).__name__, value))
        self._scheduler = value

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        logger.debug('Condition ({0}) setting owner to {1}'.format(type(self).__name__, value))
        self._owner = value

    def is_satisfied(self):
        logger.debug('Condition ({0}) using scheduler {1}'.format(type(self).__name__, self.scheduler))
        has_args = len(self.args) > 0
        has_kwargs = len(self.kwargs) > 0

        if has_args and has_kwargs:
            return self.func(self.dependencies, *self.args, **self.kwargs)
        if has_args:
            return self.func(self.dependencies, *self.args)
        if has_kwargs:
            return self.func(self.dependencies, **self.kwargs)
        return self.func(self.dependencies)

#########################################################################################################
# Included Conditions
#########################################################################################################

######################################################################
# Static Conditions
#   - independent of components and time
######################################################################


class Always(Condition):
    """Always

    Parameters:

        none

    Satisfied when:

        - always satisfied.

    """
    def __init__(self):
        super().__init__(True, lambda x: x)


class Never(Condition):
    """Never

    Parameters:

        none

    Satisfied when:

        - never satisfied.
    """
    def __init__(self):
        super().__init__(False, lambda x: x)

######################################################################
# Composite Conditions
#   - based on other Conditions
######################################################################

# TODO: create this class to subclass All and Any from
# class CompositeCondition(Condition):
    # def


class All(Condition):
    """All

    Parameters:

        args: one or more `Conditions <Condition>`

    Satisfied when:

        - all of the Conditions in args are satisfied.

    Notes:

        - To initialize with a list (for example)::

            conditions = [AfterNCalls(mechanism, 5) for mechanism in mechanism_list]

          unpack the list to supply its members as args::

           composite_condition = All(*conditions)

    """
    def __init__(self, *args):
        super().__init__(args, self.satis)

    @Condition.scheduler.setter
    def scheduler(self, value):
        for cond in self.dependencies:
            logger.debug('schedule setter: Setting scheduler of {0} to ({1})'.format(cond, value))
            if cond.scheduler is None:
                cond.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        for cond in self.dependencies:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    def satis(self, conds):
        for cond in conds:
            if not cond.is_satisfied():
                return False
        return True


class Any(Condition):
    """Any

    Parameters:

        args: one or more `Conditions <Condition>`

    Satisfied when:

        - one or more of the Conditions in **args** is satisfied.

    Notes:

        - To initialize with a list (for example)::

            conditions = [AfterNCalls(mechanism, 5) for mechanism in mechanism_list]

          unpack the list to supply its members as args::

           composite_condition = All(*conditions)

    """
    def __init__(self, *args):
        super().__init__(args, self.satis)

    @Condition.scheduler.setter
    def scheduler(self, value):
        logger.debug('Any setter args: {0}'.format(self.dependencies))
        for cond in self.dependencies:
            logger.debug('schedule setter: Setting scheduler of {0} to ({1})'.format(cond, value))
            if cond.scheduler is None:
                cond.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        for cond in self.dependencies:
            logger.debug('owner setter: Setting owner of {0} to ({1})'.format(cond, value))
            if cond.owner is None:
                cond.owner = value

    def satis(self, conds):
        for cond in conds:
            if cond.is_satisfied():
                return True
        return False


class Not(Condition):
    """Not

    Parameters:

        condition(Condition): a `Condition`

    Satisfied when:

        - the Condition is not satisfied.

    """
    def __init__(self, condition):
        super().__init__(condition, lambda c: not c.is_satisfied())

    @Condition.scheduler.setter
    def scheduler(self, value):
        self.dependencies.scheduler = value

    @Condition.owner.setter
    def owner(self, value):
        self.dependencies.owner = value

######################################################################
# Time-based Conditions
#   - satisfied based only on TimeScales
######################################################################


class BeforePass(Condition):
    """BeforePass

    Parameters:

        n(int): the 'PASS' before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\ es (default: TimeScale.TRIAL)

    Satisfied when:

        - at most n-1 `PASS`\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `PASS` is 0, the second `PASS` is 1, etc.);
          so, ``BeforePass(2)`` is satisfied at `PASS` 0 and `PASS` 1.

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] < n
        super().__init__(n, func, time_scale)


class AtPass(Condition):
    """AtPass

    Parameters:

        n(int): the `PASS` at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\ es (default: TimeScale.TRIAL)

    Satisfied when:

        - exactly n `PASS`\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first 'PASS' is pass 0, the second 'PASS' is 1, etc.);
          so, ``AtPass(1)`` is satisfied when a single `PASS` (`PASS` 0) has occurred, and ``AtPass(2) is satisfied
          when two `PASS`\ es have occurred (`PASS` 0 and `PASS` 1), etc..

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.PASS] == n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.
                                     format(type(self).__name__, e, time_scale))
        super().__init__(n, func)


class AfterPass(Condition):
    """AfterPass

    Parameters:

        n(int): the `PASS` after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\ es (default: TimeScale.TRIAL)

    Satisfied when:

        - at least n+1 `PASS`\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `PASS` is 0, the second `PASS` is 1, etc.); so,
          ``AfterPass(1)`` is satisfied after `PASS` 1 has occurred and thereafter (i.e., in `PASS`\ es 2, 3, 4, etc.).

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] > n
        super().__init__(n, func, time_scale)


class AfterNPasses(Condition):
    """AfterNPasses

    Parameters:

        n(int): the number of `PASS`\ es after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\ es (default: TimeScale.TRIAL)


    Satisfied when:

        - at least n `PASS`\ es have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] >= n
        super().__init__(n, func, time_scale)


class EveryNPasses(Condition):
    """EveryNPasses

    Parameters:

        n(int): the frequency of passes with which this condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `PASS`\ es (default: TimeScale.TRIAL)

    Satisfied when:

        - `PASS` 0;

        - the specified number of `PASS`\ es that has occurred within a unit of time (at the `TimeScale` specified by
          **time_scale**) is evenly divisible by n.

    """
    def __init__(self, n, time_scale=TimeScale.TRIAL):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.PASS] % n == 0
        super().__init__(n, func, time_scale)


class BeforeTrial(Condition):
    """BeforeTrial

    Parameters:

        n(int): the `TRIAL` before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL`\ s (default: TimeScale.RUN)

    Satisfied when:

        - at most n-1 `TRIAL`\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `TRIAL` is 0, the second `TRIAL` is 1, etc.);
          so, ``BeforeTrial(2)`` is satisfied at `TRIAL` 0 and `TRIAL` 1.

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.TRIAL] < n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.
                                     format(type(self).__name__, e, time_scale))
        super().__init__(n, func)


class AtTrial(Condition):
    """AtTrial

    Parameters:

        n(int): the `TRIAL` at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL`\ s (default: TimeScale.RUN)

    Satisfied when:

        - exactly n `TRIAL`\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:

        - Counts of TimeScales are zero-indexed (that is, the first `TRIAL` is 0, the second `TRIAL` is 1, etc.);
          so, ``AtTrial(1)`` is satisfied when one `TRIAL` (`TRIAL` 0) has already occurred.

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.TRIAL] == n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.
                                     format(type(self).__name__, e, time_scale))
        super().__init__(n, func)


class AfterTrial(Condition):
    """AfterTrial

    Parameters:

        n(int): the `TRIAL` after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL`\ s. (default: TimeScale.RUN)

    Satisfied when:

        - at least n+1 `TRIAL`\ s have occurred within one unit of time at the `TimeScale` specified by **time_scale**.

    Notes:
        - Counts of TimeScales are zero-indexed (that is, the first `TRIAL` is 0, the second `TRIAL` is 1, etc.);
          so,  ``AfterPass(1)`` is satisfied after `TRIAL` 1 has occurred and thereafter (i.e., in `TRIAL`\ s 2, 3, 4,
          etc.).

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            try:
                return self.scheduler.times[time_scale][TimeScale.TRIAL] > n
            except KeyError as e:
                raise ConditionError('{0}: {1}, is time_scale set correctly? Currently: {2}'.
                                     format(type(self).__name__, e, time_scale))
        super().__init__(n, func)


class AfterNTrials(Condition):
    """AfterNTrials

    Parameters:

        n(int): the number of `TRIAL`\ s after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting `TRIAL`\ s (default: TimeScale.RUN)

    Satisfied when:

        - at least n `TRIAL`\ s have occured  within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, n, time_scale=TimeScale.RUN):
        def func(n, time_scale):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            return self.scheduler.times[time_scale][TimeScale.TRIAL] >= n
        super().__init__(n, func, time_scale)

######################################################################
# Component-based Conditions
#   - satisfied based on executions or state of Components
######################################################################


class BeforeNCalls(Condition):
    """BeforeNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** before which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed at most n times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls < n
        super().__init__(dependency, func, n)

# NOTE:
# The behavior of AtNCalls is not desired (i.e. depending on the order mechanisms are checked, B running AtNCalls(A, x))
# may run on both the xth and x+1st call of A; if A and B are not parent-child
# A fix could invalidate key assumptions and affect many other conditions
# Since this condition is unlikely to be used, it's best to leave it for now


class AtNCalls(Condition):
    """AtNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** at which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed exactly n times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls == n
        super().__init__(dependency, func, n)


class AfterCall(Condition):
    """AfterCall

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed at least n+1 times
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls > n
        super().__init__(dependency, func, n)


class AfterNCalls(Condition):
    """AfterNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the number of executions of **component** after which the Condition is satisfied

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)

    Satisfied when:

        - the Component specified in **component** has executed at least n times
          within one unit of time at the `TimeScale` specified by **time_scale**.


    """
    def __init__(self, dependency, n, time_scale=TimeScale.TRIAL):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            num_calls = self.scheduler.counts_total[time_scale][dependency]
            logger.debug('{0} has reached {1} num_calls in {2}'.format(dependency, num_calls, time_scale.name))
            return num_calls >= n
        super().__init__(dependency, func, n)


class AfterNCallsCombined(Condition):
    """
    AfterNCallsCombined

    Parameters:
        - *components (Components): variable length
        - n (int): the number of executions of all components after which this condition is satisfied.
          Defaults to None
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of components.
          Defaults to TimeScale.TRIAL


    Parameters:

        components(Components):  an iterable of Components on which the Condition depends

        n(int): the number of combined executions of all Components specified in **components** after which the \
        Condition is satisfied (default: None)

        time_scale(TimeScale): the TimeScale used as basis for counting executions of **component** \
        (default: TimeScale.TRIAL)


    Satisfied when:

        - there have been at least n+1 executions among all of the Components specified in **components**
          within one unit of time at the `TimeScale` specified by **time_scale**.

    """
    def __init__(self, *dependencies, n=None, time_scale=TimeScale.TRIAL):
        logger.debug('{0} args: deps {1}, n {2}, ts {3}'.format(type(self).__name__, dependencies, n, time_scale))

        def func(_none, *dependencies, n=None):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            if n is None:
                raise ConditionError('{0}: keyword argument n is None'.format(type(self).__name__))
            count_sum = 0
            for d in dependencies:
                count_sum += self.scheduler.counts_total[time_scale][d]
                logger.debug('{0} has reached {1} num_calls in {2}'.
                             format(d, self.scheduler.counts_total[time_scale][d], time_scale.name))
            return count_sum >= n
        super().__init__(None, func, *dependencies, n=n)


class EveryNCalls(Condition):
    """EveryNCalls

    Parameters:

        component(Component):  the Component on which the Condition depends

        n(int): the frequency of executions of **component** at which the Condition is satisfied


    Satisfied when:

        - since the last time this condition's owner was called, the number of calls of **component** is at least n

        COMMENT:
            JDC: IS THE FOLLOWING TRUE OF ALL OF THE ABOVE AS WELL??
            K: No, EveryNCalls is tricky in how it needs to be implemented, because it's in a sense
            tracking the relative frequency of calls between two objects. So the idea is that the scheduler
            tracks how many executions of a component are "useable" by other components for EveryNCalls conditions.
            So, suppose you had something like add_condition(B, All(AfterNCalls(A, 10), EveryNCalls(A, 2))). You
            would want the AAB pattern to start happening after A has run 10 times. Useable counts allows B to see
            whether A has run enough times for it to run, and then B spends its "useable executions" of A. Then,
            A must run two more times for B to run again. If you didn't reset the counts of A useable by B
            to 0 (question below) when B runs, then in the
            above case B would continue to run every pass for the next 4 passes, because it would see an additional
            8 executions of A it could spend to execute.
        COMMENT
          since the owner of the Condition has executed.

    COMMENT:
        JDC: DON'T UNDERSTAND THE FOLLOWING:
    COMMENT
    Notes:
        Whenever a Component is run, the Scheduler's count of each dependency that is "useable" by the Component is
        reset to 0

    """
    def __init__(self, dependency, n):
        def func(dependency, n):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            num_calls = self.scheduler.counts_useable[dependency][self.owner]
            logger.debug('{0} has reached {1} num_calls'.format(dependency, num_calls))
            return num_calls >= n
        super().__init__(dependency, func, n)


class JustRan(Condition):
    """
    JustRan

    Parameters:
        - dependency (Component):

    Satisfied when:
        - dependency has been run (or told to run) in the previous TimeScale.TIME_STEP

    Notes:
        This condition can transcend divisions between TimeScales. That is, if A runs in the final time step in a trial,
        JustRan(A) is satisfied at the beginning of the next trial.

    """
    def __init__(self, dependency):
        def func(dependency):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            logger.debug('checking if {0} in previous execution step set'.format(dependency))
            try:
                return dependency in self.scheduler.execution_list[-1]
            except TypeError:
                return dependency == self.scheduler.execution_list[-1]
        super().__init__(dependency, func)


class AllHaveRun(Condition):
    """
    AllHaveRun

    Parameters:
        - *dependencies (Components): variable length
        - time_scale (TimeScale): the TimeScale used as basis for counting executions of dependencies.
          Defaults to TimeScale.TRIAL

    Satisfied when:
        - All dependencies have been executed at least 1 time within the scope of time_scale

    Notes:

    """
    def __init__(self, *dependencies, time_scale=TimeScale.TRIAL):
        def func(_none, *dependencies):
            if self.scheduler is None:
                raise ConditionError('{0}: self.scheduler is None - scheduler must be assigned'.
                                     format(type(self).__name__))
            if len(dependencies) == 0:
                dependencies = self.scheduler.nodes
            for d in dependencies:
                if self.scheduler.counts_total[time_scale][d] < 1:
                    return False
            return True
        super().__init__(None, func, *dependencies)


class WhenFinished(Condition):
    """
    WhenFinished

    Parameters:
        - dependency (Component):

    Satisfied when:
        - dependency has "finished" (i.e. its is_finished attribute is True)

    Notes:
        This is a dynamic condition.
        The is_finished concept varies among components, and is currently implemented in:
            `DDM`<DDM>

    """
    def __init__(self, dependency):
        def func(dependency):
            try:
                return dependency.is_finished
            except AttributeError as e:
                raise ConditionError('WhenFinished: Unsupported dependency type: {0}; ({1})'.
                                     format(type(dependency), e))

        super().__init__(dependency, func)


class WhenFinishedAny(Condition):
    """
    WhenFinishedAny

    Parameters:
        - *dependencies (Components): variable length

    Satisfied when:
        - any of the dependencies have "finished" (i.e. its is_finished attribute is True)

    Notes:
    This is a dynamic condition.
        This is a convenience class; WhenFinishedAny(A, B, C) is equivalent to
        Any(WhenFinished(A), WhenFinished(B), WhenFinished(C))
        If no dependencies are specified, the condition will default to checking all of its scheduler's Components.
        The is_finished concept varies among components, and is currently implemented in:
            `DDM`<DDM>

    """
    def __init__(self, *dependencies):
        def func(_none, *dependencies):
            if len(dependencies) == 0:
                dependencies = self.scheduler.nodes
            for d in dependencies:
                try:
                    if d.is_finished:
                        return True
                except AttributeError as e:
                    raise ConditionError('WhenFinishedAny: Unsupported dependency type: {0}; ({1})'.format(type(d), e))
            return False

        super().__init__(None, func, *dependencies)


class WhenFinishedAll(Condition):
    """
    WhenFinishedAll

    Parameters:
        - *dependencies (Components): variable length

    Satisfied when:
        - all of the dependencies have "finished" (i.e. its is_finished attribute is True)

    Notes:
        This is a dynamic condition.
        This is a convenience class; WhenFinishedAll(A, B, C) is equivalent to
        All(WhenFinished(A), WhenFinished(B), WhenFinished(C))
        If no dependencies are specified, the condition will default to checking all of its scheduler's Components.
        The is_finished concept varies among components, and is currently implemented in:
            `DDM`<DDM>

    """
    def __init__(self, *dependencies):
        def func(_none, *dependencies):
            if len(dependencies) == 0:
                dependencies = self.scheduler.nodes
            for d in dependencies:
                try:
                    if not d.is_finished:
                        return False
                except AttributeError as e:
                    raise ConditionError('WhenFinishedAll: Unsupported dependency type: {0}; ({1})'.format(type(d), e))
            return True

        super().__init__(None, func, *dependencies)
