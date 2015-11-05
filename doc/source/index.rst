.. lda documentation master file, created by
   sphinx-quickstart on Tue Jul  9 22:26:36 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lda: Topic modeling with latent Dirichlet Allocation
====================================================

.. raw:: html

    <div class="wy-text-large">

**lda** implements latent Dirichlet allocation (LDA) using collapsed Gibbs
sampling. lda is fast and can be installed without a compiler on Linux, OS X,
and Windows.

.. raw:: html

    </div>

The interface follows conventions found in scikit-learn_.  The following
demonstrates how to inspect a model of a subset of the Reuters news dataset.
(The input below, ``X``, is a document-term matrix.)

.. code-block:: python

    >>> import numpy as np
    >>> import lda
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
    ...     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    ...     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    Topic 0: british churchill sale million major letters west
    Topic 1: church government political country state people party
    Topic 2: elvis king fans presley life concert young
    Topic 3: yeltsin russian russia president kremlin moscow michael
    Topic 4: pope vatican paul john surgery hospital pontiff
    Topic 5: family funeral police miami versace cunanan city
    Topic 6: simpson former years court president wife south
    Topic 7: order mother successor election nuns church nirmala
    Topic 8: charles prince diana royal king queen parker
    Topic 9: film french france against bardot paris poster
    Topic 10: germany german war nazi letter christian book
    Topic 11: east peace prize award timor quebec belo
    Topic 12: n't life show told very love television
    Topic 13: years year time last church world people
    Topic 14: mother teresa heart calcutta charity nun hospital
    Topic 15: city salonika capital buddhist cultural vietnam byzantine
    Topic 16: music tour opera singer israel people film
    Topic 17: church catholic bernardin cardinal bishop wright death
    Topic 18: harriman clinton u.s ambassador paris president churchill
    Topic 19: city museum art exhibition century million churches

Contents:

.. toctree::
   :maxdepth: 2

   getting_started
   installation
   api
   contributing
   groups
   whats_new

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _scikit-learn: http://scikit-learn.org
