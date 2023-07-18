# coding=utf-8
from __future__ import absolute_import, unicode_literals  # noqa
import os

import numpy as np
import oslotest.base
import scipy.sparse
import scipy.stats

import lda
import lda.utils


class TestLDATransform(oslotest.base.BaseTestCase):

    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(__file__)
        reuters_ldac_fn = os.path.join(test_dir, 'reuters.ldac')
        cls.dtm = dtm = lda.utils.ldac2dtm(open(reuters_ldac_fn), offset=0)
        cls.dtm_sparse = scipy.sparse.csr_matrix(dtm)
        cls.n_iter = n_iter = 400
        cls.n_topics = n_topics = 15
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

        n_docs = 10
        assert n_docs < len(dtm) / 2
        dtm_test = dtm[:n_docs]
        doc_topic_test_true = doc_topic[:n_docs]
        doc_topic_test = model.transform(dtm_test)

        S = 2000
        kl_div_dist = np.empty(S)
        np.random.seed(random_seed)
        for s in range(S):
            # scipy.stats.entropy(p, q) calculates Kullback-Leibler divergence
            kl_div_dist[s] = scipy.stats.entropy(doc_topic_test_true[np.random.choice(len(doc_topic_test_true))],
                                                 doc_topic[np.random.choice(len(doc_topic))])
        quantiles = scipy.stats.mstats.mquantiles(kl_div_dist, prob=np.linspace(0, 1, 500, endpoint=False))

        for p, q in zip(doc_topic_test_true, doc_topic_test):
            kl_div = scipy.stats.entropy(p, q)
            quantile = np.searchsorted(quantiles, kl_div) / len(quantiles)
            self.assertLess(quantile, 0.05)

    def test_lda_transform_basic(self):
        """Basic checks on transform"""
        model = self.model
        dtm = self.dtm

        n_docs = 3
        n_topics = len(model.components_)
        dtm_test = dtm[0:n_docs]
        doc_topic_test = model.transform(dtm_test)
        self.assertEqual(doc_topic_test.shape, (n_docs, n_topics))
        np.testing.assert_array_almost_equal(doc_topic_test.sum(axis=1), 1)

        # one document
        dtm_test = dtm[0]
        doc_topic_test = model.transform(dtm_test)
        self.assertEqual(doc_topic_test.shape, (1, n_topics))
        np.testing.assert_array_almost_equal(doc_topic_test.sum(axis=1), 1)

    def test_lda_transform_basic_sparse(self):
        """Basic checks on transform"""
        model = self.model
        dtm = self.dtm_sparse

        n_docs = 3
        n_topics = len(model.components_)
        dtm_test = dtm[0:n_docs]
        doc_topic_test = model.transform(dtm_test)
        self.assertEqual(doc_topic_test.shape, (n_docs, n_topics))
        np.testing.assert_array_almost_equal(doc_topic_test.sum(axis=1), 1)

        # one document
        dtm_test = dtm[0]
        doc_topic_test = model.transform(dtm_test)
        self.assertEqual(doc_topic_test.shape, (1, n_topics))
        np.testing.assert_array_almost_equal(doc_topic_test.sum(axis=1), 1)
