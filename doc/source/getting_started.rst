===============
Getting started
===============

The following demonstrates how to explore a model of a subset of the Reuters
news dataset.

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
    >>> model.fit(X)
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
