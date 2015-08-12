===============
Style Guidlines
===============
Before contributing a patch, please read the Python "Style Commandments" written
by the OpenStack developers: http://docs.openstack.org/developer/hacking/

========================
Building in Develop Mode
========================

To build in develop mode on OS X, first install Cython and pbr. Then run::

  git clone https://github.com/ariddell/lda.git
  cd lda
  make cython
  python setup.py develop
  
