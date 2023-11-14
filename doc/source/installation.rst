.. _installation-instructions:

==============
Installing lda
==============

lda requires Python (>=3.9,<=3.12) and NumPy (>=1.13.0,<2.0). If these
requirements are satisfied, lda should install successfully on Linux, macOS and Windows with::

    pip install lda


Installation from source
------------------------

Installing from source requires you to have installed the Python development
headers and a working C/C++ compiler. Under Debian-based operating systems,
which include Ubuntu, you can install all these requirements by issuing::

    sudo apt-get install build-essential python3-dev python3-setuptools \
                         python3-numpy

You can compile and install lda using the ``pip install`` or ``poetry install`` command.
