===============
Installing ANML
===============

Overview
--------

The package anml (pronounced "animal") is written in Python
and requires Python 3.7 in order to use `dataclasses`.

Installation from Source
------------------------

To install anml, git clone, and then use pip install.

::

    git clone https://github.com/ihmeuw-msca/anml.git
    cd anml
    pip install .

For Developers
--------------

To install anml in development mode,

::

    pip install -e .[test,docs]
