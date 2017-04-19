lda: Topic modeling with latent Dirichlet allocation
====================================================

|pypi| |travis| |zenodo|

``lda`` implements latent Dirichlet allocation (LDA) using collapsed Gibbs
sampling. ``lda`` is fast and is tested on Linux, OS X, and Windows.

You can read more about lda in `the documentation <http://pythonhosted.org/lda>`_.

Installation
------------

``pip install lda``

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
    >>> X.sum()
    84010
    >>> model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
    >>> model.fit(X)  # model.fit_transform(X) is also available
    >>> topic_word = model.topic_word_  # model.components_ also works
    >>> n_top_words = 8
    >>> for i, topic_dist in enumerate(topic_word):
    ...     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    ...     print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    Topic 0: british churchill sale million major letters west britain
    Topic 1: church government political country state people party against
    Topic 2: elvis king fans presley life concert young death
    Topic 3: yeltsin russian russia president kremlin moscow michael operation
    Topic 4: pope vatican paul john surgery hospital pontiff rome
    Topic 5: family funeral police miami versace cunanan city service
    Topic 6: simpson former years court president wife south church
    Topic 7: order mother successor election nuns church nirmala head
    Topic 8: charles prince diana royal king queen parker bowles
    Topic 9: film french france against bardot paris poster animal
    Topic 10: germany german war nazi letter christian book jews
    Topic 11: east peace prize award timor quebec belo leader
    Topic 12: n't life show told very love television father
    Topic 13: years year time last church world people say
    Topic 14: mother teresa heart calcutta charity nun hospital missionaries
    Topic 15: city salonika capital buddhist cultural vietnam byzantine show
    Topic 16: music tour opera singer israel people film israeli
    Topic 17: church catholic bernardin cardinal bishop wright death cancer
    Topic 18: harriman clinton u.s ambassador paris president churchill france
    Topic 19: city museum art exhibition century million churches set

The document-topic distributions are available in ``model.doc_topic_``.

.. code-block:: python

    >>> doc_topic = model.doc_topic_
    >>> for i in range(10):
    ...     print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
    0 UK: Prince Charles spearheads British royal revolution. LONDON 1996-08-20 (top topic: 8)
    1 GERMANY: Historic Dresden church rising from WW2 ashes. DRESDEN, Germany 1996-08-21 (top topic: 13)
    2 INDIA: Mother Teresa's condition said still unstable. CALCUTTA 1996-08-23 (top topic: 14)
    3 UK: Palace warns British weekly over Charles pictures. LONDON 1996-08-25 (top topic: 8)
    4 INDIA: Mother Teresa, slightly stronger, blesses nuns. CALCUTTA 1996-08-25 (top topic: 14)
    5 INDIA: Mother Teresa's condition unchanged, thousands pray. CALCUTTA 1996-08-25 (top topic: 14)
    6 INDIA: Mother Teresa shows signs of strength, blesses nuns. CALCUTTA 1996-08-26 (top topic: 14)
    7 INDIA: Mother Teresa's condition improves, many pray. CALCUTTA, India 1996-08-25 (top topic: 14)
    8 INDIA: Mother Teresa improves, nuns pray for "miracle". CALCUTTA 1996-08-26 (top topic: 14)
    9 UK: Charles under fire over prospect of Queen Camilla. LONDON 1996-08-26 (top topic: 8)


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

Other implementations
---------------------
- scikit-learn_'s `LatentDirichletAllocation <http://scikit-learn.org/dev/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html>`_ (uses online variational inference)
- `gensim <https://pypi.python.org/pypi/gensim>`_ (uses online variational inference)

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
.. _Pritchard et al. (2000): http://www.genetics.org/content/155/2/945.full
.. _Griffiths and Steyvers (2004): http://www.pnas.org/content/101/suppl_1/5228.abstract

.. |pypi| image:: https://badge.fury.io/py/lda.png
    :target: https://pypi.python.org/pypi/lda
    :alt: pypi version

.. |travis| image:: https://travis-ci.org/ariddell/lda.png?branch=master
    :target: https://travis-ci.org/ariddell/lda
    :alt: travis-ci build status

.. |zenodo| image:: https://zenodo.org/badge/doi/10.5281/zenodo.57927.svg
    :target: http://dx.doi.org/10.5281/zenodo.57927
    :alt: Zenodo citation
