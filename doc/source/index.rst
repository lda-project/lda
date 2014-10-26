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

The interface follows conventions found in scikit-learn_. Here's an example:

.. code-block:: python

    >>> import numpy as np
    >>> import lda
    >>> import lda.datasets
    >>> X = lda.datasets.load_reuters()
    >>> model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    >>> doc_topic = model.fit_transform(X)
    >>> topic_word = model.topic_word_  # model.components_ also works
    >>> for i, topic_dist in enumerate(topic_word):
    ...     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-8:-1]
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


Contents:

.. toctree::
   :maxdepth: 2

   getting_started
   installation
   api
   contributing
   groups

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _scikit-learn: http://scikit-learn.org
