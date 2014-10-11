.. _installation-instructions:

==============
Installing lda
==============

lda requires Python (>= 2.7 or >= 3.3) and NumPy (>= 1.6.1). If these
requirements are satisfied, lda should install successfully with::

    pip install lda

If you encounter problems, consult the platform-specific instructions below.

Windows
-------

First you need to install `numpy <http://numpy.scipy.org/>`_ from the official
installer.

.. FIXME: update this when Numpy has Windows wheels available

Wheel packages (.whl files) for lda from `PyPI
<https://pypi.python.org/pypi/lda>`_ can be installed with the `pip
<http://pip.readthedocs.org/en/latest/installing.html>`_ utility.  Open
a console and type the following to install lda::

    pip install lda

.. FIXME: remove the following when Python 3.3 is no longer widely used

In order to use wheels, you will need to have pip version 1.4 or higher and
setuptools version 0.8 or higher.

Mac OS X
--------

lda and its dependencies are all available as wheel packages for Mac OS X::

    pip install numpy lda

.. FIXME: remove the following when Python 3.3 is no longer widely used

In order to use wheels, you will need to have pip version 1.4 or higher and
setuptools version 0.8 or higher.

Linux
-----

If you are running an Ubuntu-based distribution of Linux, you may install lda
using a PPA::

    sudo add-apt-repository ppa:ariddell/lda
    sudo apt-get update
    sudo apt-get install python3-lda  # or, python-lda for the Python 2.7 version

Alternatively, you may install lda from source. In order for this to work,
however, you need to install a compiler and associated build dependencies.

Installing build dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing from source requires you to have installed the Python development
headers and a working C/C++ compiler.  Under Debian-based operating systems,
which include Ubuntu, if you have Python 2 you can install all these
requirements by issuing::

    sudo apt-get install build-essential python-dev python-setuptools \
                         python-numpy

If you have Python 3::

    sudo apt-get install build-essential python3-dev python3-setuptools \
                         python3-numpy

With these installed, you may install lda::


    $ pip install lda


Or, if you have virtualenvwrapper installed::

    $ mkvirtualenv lda
    $ pip install lda
