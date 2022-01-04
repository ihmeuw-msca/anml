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

    pip install anml

For Developers
--------------

To install anml in development mode,

::

    git clone https://github.com/ihmeuw-msca/anml.git
    cd anml
    pip install -e .[test,docs]
