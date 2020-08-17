.. _installation-instructions:

==============
Installing lda
==============

lda requires Python (>= 3.6) and NumPy (>= 1.13.0). If these
requirements are satisfied, lda should install successfully on Linux and macOS with::

    pip install lda

If you encounter problems, consult the platform-specific instructions below.

Mac OS X
--------

lda and its dependencies are all available as wheel packages for Mac OS X::

    pip install lda

Linux
-----

lda and its dependencies are all available as wheel packages for most distributions of Linux::

    pip install lda

Windows
-------

lda must be built from source on Windows. There are no wheels at this time.

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
