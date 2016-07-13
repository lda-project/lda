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

lda and its dependencies are all available as wheel packages for Windows::

    pip install lda

Mac OS X
--------

lda and its dependencies are all available as wheel packages for Mac OS X::

    pip install lda

Linux
-----

lda and its dependencies are all available as wheel packages for most distributions of Linux::

    pip install lda

Installation from source
------------------------

Installing from source requires you to have installed the Python development
headers and a working C/C++ compiler. Under Debian-based operating systems,
which include Ubuntu, you can install all these requirements by issuing::

    sudo apt-get install build-essential python3-dev python3-setuptools \
                         python3-numpy

Before attempting a command such as ``python setup.py install`` you will need to run
Cython to generate the relevant C files::

    make cython
