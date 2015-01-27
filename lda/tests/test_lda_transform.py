# coding=utf-8
from __future__ import absolute_import, unicode_literals  # noqa
import os

import numpy as np
import oslotest.base
import scipy.stats

import lda
import lda.utils


class TestLDATransform(oslotest.base.BaseTestCase):

    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(__file__)
        reuters_ldac_fn = os.path.join(test_dir, 'reuters.ldac')
        cls.dtm = dtm = lda.utils.ldac2dtm(open(reuters_ldac_fn), offset=0)
        # TODO: boost n_iter when testing done
        cls.n_iter = n_iter = 100
        cls.n_topics = n_topics = 10
        cls.random_seed = random_seed = 1
        cls.model = model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_seed)
        cls.doc_topic = model.fit_transform(dtm)

    def test_lda_transform_null(self):
        """Evaluate transform by checking predicted doc_topic distribution

        In this case, our null hypothesis is that we are doing no better than
        picking at random from a fitted model and calculating the KL divergence.
        """
        random_seed = self.random_seed
        model = self.model
        dtm = self.dtm
        doc_topic = self.doc_topic

        # TODO: test several documents
        test_index = 0
        dtm_test = dtm[test_index]
        doc_topic_test_true = doc_topic[test_index]
        doc_topic_test = np.atleast_2d(model.transform(dtm_test))[0]
        self.assertAlmostEqual(sum(doc_topic_test), 1)

        S = 2000
        kl_div_dist = np.empty(S)
        np.random.seed(random_seed)
        for s in range(S):
            q = doc_topic[np.random.choice(len(doc_topic))]
            # scipy.stats.entropy(p, q) calculates Kullback-Leibler divergence
            kl_div_dist[s] = scipy.stats.entropy(doc_topic_test_true, q)
        quantiles = scipy.stats.mstats.mquantiles(kl_div_dist, prob=np.linspace(0, 1, 100, endpoint=False))
        kl_div = scipy.stats.entropy(doc_topic_test_true, doc_topic_test)
        quantile = np.searchsorted(quantiles, kl_div) / len(quantiles)
        self.assertLessEqual(quantile, 0.01)

    def test_lda_transform_basic(self):
        """Basic checks on transform"""
        random_seed = self.random_seed
        model = self.model
        dtm = self.dtm
        doc_topic = self.doc_topic

        n_docs = 3
        n_topics = len(model.components_)
        dtm_test = dtm[0:n_docs]
        doc_topic_test = model.transform(dtm_test)
        self.assertEqual(doc_topic_test.shape, (n_docs, n_topics))
        dtm_test = dtm[0]
        doc_topic_test = model.transform(dtm_test)
        self.assertEqual(doc_topic_test.shape, (1, n_topics))
