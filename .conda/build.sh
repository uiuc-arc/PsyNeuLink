#!/usr/bin/env bash

# Install anything that didn't get conda installed via pip.
# We need to turn pip index back on because Anaconda turns
# it off for some reason. Just pip install -r requirements.txt
# doesn't seem to work, tensorflow-gpu, jsonpickle, networkx,
# all get installed twice if we do this. pip doesn't see the
# conda install of the packages.

# Install the pip dependencies and their dependencies. Conda turns of
# pip index and dependencies by default so re-enable them. Had to figure
# this out myself, ughhh.
export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False
pip install typecheck-decorator==1.2 toposort==1.4

# Install requires setuptools-scm
pip install setuptools-scm

# Install psyneulink itself.
# NOTE: This is the recommended way to install packages
python setup.py install --single-version-externally-managed --record=record.txt
