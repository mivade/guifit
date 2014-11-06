guifit
======

A simple graphical interface for doing quick fits of simple data based
on the guiqwt's fit widget.

Interactive usage
-----------------

To fit data, simply run ``python guidata.py`` and select a data file
and fit function to use.

Defining fit functions
----------------------

Fit functions are defined through classes derived from the
``FitFunction`` class. Only two methods need to be defined:
``__init__`` should store the x and y data and setup fit parameters,
while ``func`` should define the actual fit function itself.

Requirements
------------

 * guiqwt_
 * guidata_

.. _guiqwt: https://code.google.com/p/guiqwt/
.. _guidata: https://code.google.com/p/guidata/
