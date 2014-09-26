lda: Topic modeling with latent Dirichlet allocation
====================================================

|pypi| |travis| |crate|

``lda`` implements latent Dirichlet allocation (LDA) using collapsed Gibbs
sampling. ``lda`` is fast and is tested on Linux, OS X, and Windows.

Installation
------------

If you have NumPy installed,

``pip install lda``

Installation does not require a compiler on Windows or OS X.

Getting started
---------------

``lda.LDA`` implements latent Dirichlet allocation (LDA). The interface follows
conventions found in scikit-learn_.

The following demonstrates how to inspect a model of a subset of the Reuters
news dataset. The input below, ``X``, is a document-term matrix (sparse matrices
are accepted).

.. code-block:: python

    >>> import numpy as np
    >>> import lda
    >>> import lda.datasets
    >>> X = lda.datasets.load_reuters()
    >>> vocab = lda.datasets.load_reuters_vocab()
    >>> titles = lda.datasets.load_reuters_titles()
    >>> X.shape
    (395, 4258)
    >>> model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    >>> model.fit(X)  # model.fit_transform(X) is also available
    >>> topic_word = model.topic_word_  # model.components_ also works
    >>> n_top_words = 8
    >>> for i, topic_dist in enumerate(topic_word):
    ...     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    ...     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    Topic 0: church people told years last year time
    Topic 1: elvis music fans york show concert king
    Topic 2: pope trip mass vatican poland health john
    Topic 3: film french against france festival magazine quebec
    Topic 4: king michael romania president first service romanian
    Topic 5: police family versace miami cunanan west home
    Topic 6: germany german war political government minister nazi
    Topic 7: harriman u.s clinton churchill ambassador paris british
    Topic 8: yeltsin russian russia president kremlin moscow operation
    Topic 9: prince queen bowles church king royal public
    Topic 10: simpson million years south irish churches says
    Topic 11: charles diana parker camilla marriage family royal
    Topic 12: east peace prize president award catholic timor
    Topic 13: order nuns india successor election roman sister
    Topic 14: pope vatican hospital surgery rome roman doctors
    Topic 15: mother teresa heart calcutta missionaries hospital charity
    Topic 16: bernardin cardinal cancer church life catholic chicago
    Topic 17: died funeral church city death buddhist israel
    Topic 18: museum kennedy cultural city culture greek byzantine
    Topic 19: art exhibition century city tour works madonna

The document-topic distributions are available in ``model.doc_topic_``.

.. code-block:: python

    >>> doc_topic = model.doc_topic_
    >>> for i in range(10):
    ...     print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
    0 UK: Prince Charles spearheads British royal revolution. LONDON 1996-08-20 (top topic: 11)
    1 GERMANY: Historic Dresden church rising from WW2 ashes. DRESDEN, Germany 1996-08-21 (top topic: 0)
    2 INDIA: Mother Teresa's condition said still unstable. CALCUTTA 1996-08-23 (top topic: 15)
    3 UK: Palace warns British weekly over Charles pictures. LONDON 1996-08-25 (top topic: 11)
    4 INDIA: Mother Teresa, slightly stronger, blesses nuns. CALCUTTA 1996-08-25 (top topic: 15)
    5 INDIA: Mother Teresa's condition unchanged, thousands pray. CALCUTTA 1996-08-25 (top topic: 15)
    6 INDIA: Mother Teresa shows signs of strength, blesses nuns. CALCUTTA 1996-08-26 (top topic: 15)
    7 INDIA: Mother Teresa's condition improves, many pray. CALCUTTA, India 1996-08-25 (top topic: 15)
    8 INDIA: Mother Teresa improves, nuns pray for "miracle". CALCUTTA 1996-08-26 (top topic: 15)
    9 UK: Charles under fire over prospect of Queen Camilla. LONDON 1996-08-26 (top topic: 0)

Requirements
------------

Python 2.7 or Python 3.3+ is required. The following packages are required

- numpy_
- pbr_

Caveat
------

``lda`` aims for simplicity. (It happens to be fast, as essential parts are
written in C via Cython_.) If you are working with a very large corpus you may
wish to use more sophisticated topic models such as those implemented in hca_
and MALLET_.  hca_ is written entirely in C and MALLET_ is written in Java.
Unlike ``lda``, hca_ can use more than one processor at a time. Both MALLET_ and
hca_ implement topic models known to be more robust than standard latent
Dirichlet allocation.

Notes
-----

Latent Dirichlet allocation is described in `Blei et al. (2003)`_ and `Pritchard
et al. (2000)`_. Inference using collapsed Gibbs sampling is described in
`Griffiths and Steyvers (2004)`_.

Important links
---------------

- Documentation: http://pythonhosted.org/lda
- Source code: https://github.com/ariddell/lda/
- Issue tracker: https://github.com/ariddell/lda/issues

Similar projects
----------------
- `z-label LDA <http://pages.cs.wisc.edu/~andrzeje/research/zl_lda.html>`_
- `gensim <https://pypi.python.org/pypi/gensim>`_

License
-------

lda is licensed under Version 2.0 of the Mozilla Public License.

.. _Python: http://www.python.org/
.. _scikit-learn: http://scikit-learn.org
.. _hca: http://www.mloss.org/software/view/527/
.. _MALLET: http://mallet.cs.umass.edu/
.. _numpy: http://www.numpy.org/
.. _pbr: https://pypi.python.org/pypi/pbr
.. _Cython: http://cython.org
.. _Blei et al. (2003): http://jmlr.org/papers/v3/blei03a.html
.. _Pritchard et al. (2000): http://www.genetics.org/content/164/4/1567.full
.. _Griffiths and Steyvers (2004): http://www.pnas.org/content/101/suppl_1/5228.abstract

.. |pypi| image:: https://badge.fury.io/py/lda.png
    :target: https://badge.fury.io/py/lda
    :alt: pypi version

.. |travis| image:: https://travis-ci.org/ariddell/lda.png?branch=master
    :target: https://travis-ci.org/ariddell/lda
    :alt: travis-ci build status

.. |crate| image:: https://pypip.in/d/lda/badge.png
    :target: https://pypi.python.org/pypi/lda
    :alt: pypi download statistics
