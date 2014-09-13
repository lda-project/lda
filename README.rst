lda: Topic modeling with latent Dirichlet allocation
====================================================

|pypi| |travis| |crate|

Topic modeling with latent Dirichlet allocation. ``lda`` aims for simplicity.

``lda`` implements latent Dirichlet allocation (LDA) using collapsed Gibbs
sampling. LDA is described in `Blei et al. (2003)`_ and `Pritchard et al. (2000)`_.

Installation
------------

``pip install lda``

Getting started
---------------

``lda.LDA`` implements latent Dirichlet allocation (LDA). The interface follows
conventions found in scikit-learn_.

.. code-block:: python

    >>> import numpy as np
    >>> import lda
    >>> X = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> model = lda.LDA(n_topics=2, n_iter=100)
    >>> doc_topic = model.fit_transform(X)  # estimate of document-topic distributions
    >>> model.components_  # estimate of topic-word distributions; model.doc_topic_ is an alias

Requirements
------------

Python 2.7 or Python 3.3+ is required. The following packages are required

- numpy_
- scipy_
- pbr_

Caveat
------

``lda`` aims for simplicity. (It happens to be fast, as essential parts are
written in C via Cython_.) If you are working with a very large corpus you may
wish to use more sophisticated topic models such as those implemented in hca_
and MALLET_.  hca_ is written in C and MALLET_ is written in Java.  Unlike
``lda``, hca_ can use more than one processor at a time.

Important links
---------------

- Documentation: http://pythonhosted.org/lda
- Source code: https://github.com/ariddell/lda/
- Issue tracker: https://github.com/ariddell/lda/issues

License
-------

lda is licensed under Version 2.0 of the Mozilla Public License.

.. _Python: http://www.python.org/
.. _scikit-learn: http://scikit-learn.org
.. _hca: http://www.mloss.org/software/view/527/
.. _MALLET: http://mallet.cs.umass.edu/
.. _numpy: http://www.numpy.org/
.. _scipy:  http://docs.scipy.org/doc/
.. _pbr: https://pypi.python.org/pypi/pbr
.. _Blei et al. (2003): http://jmlr.org/papers/v3/blei03a.html
.. _Pritchard et al. (2000): http://www.genetics.org/content/164/4/1567.full


.. |pypi| image:: https://badge.fury.io/py/lda.png
    :target: https://badge.fury.io/py/lda
    :alt: pypi version

.. |travis| image:: https://travis-ci.org/ariddell/lda.png?branch=master
    :target: https://travis-ci.org/ariddell/lda
    :alt: travis-ci build status

.. |crate| image:: https://pypip.in/d/lda/badge.png
    :target: https://pypi.python.org/pypi/lda
    :alt: pypi download statistics
