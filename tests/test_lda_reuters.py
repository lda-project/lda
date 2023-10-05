# coding=utf-8
from __future__ import absolute_import, unicode_literals  # noqa
import os

import numpy as np
import oslotest.base

import lda
import lda.utils


class TestLDANewsReuters(oslotest.base.BaseTestCase):

    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(__file__)
        reuters_ldac_fn = os.path.join(test_dir, 'reuters.ldac')
        cls.dtm = dtm = lda.utils.ldac2dtm(open(reuters_ldac_fn), offset=0)
        cls.n_iter = n_iter = 1
        cls.n_topics = n_topics = 10
        cls.random_seed = random_seed = 1
        cls.model = model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_seed)
        cls.doc_topic = model.fit_transform(dtm)

    def test_lda_news(self):
        dtm = self.dtm
        doc_topic = self.doc_topic
        self.assertEqual(len(doc_topic), len(dtm))

    def test_lda_attributes(self):
        dtm = self.dtm
        doc_topic = self.doc_topic
        model = self.model

        # check dims
        N = dtm.sum()
        D, V = dtm.shape
        _, K = doc_topic.shape
        self.assertEqual(model.doc_topic_.shape, doc_topic.shape)
        np.testing.assert_array_equal(model.doc_topic_, doc_topic)
        self.assertEqual(model.doc_topic_.shape, (D, K))
        self.assertEqual(model.ndz_.shape, (D, K))
        self.assertEqual(model.topic_word_.shape, (K, V))
        self.assertEqual(model.nzw_.shape, (K, V))

        # check contents
        self.assertAlmostEqual(model.nzw_.sum(), N)
        self.assertAlmostEqual(model.ndz_.sum(), N)
        self.assertAlmostEqual(model.nz_.sum(), N)
        self.assertAlmostEqual(model.doc_topic_.sum(), D)
        self.assertAlmostEqual(model.topic_word_.sum(), K)
        np.testing.assert_array_equal(model.ndz_.sum(axis=0), model.nz_)

        # check distributions sum to one
        np.testing.assert_array_almost_equal(model.doc_topic_.sum(axis=1), np.ones(D))
        np.testing.assert_array_almost_equal(model.topic_word_.sum(axis=1), np.ones(K))

    def test_lda_random_seed(self):
        dtm = self.dtm
        doc_topic = self.doc_topic
        n_iter = self.n_iter
        n_topics = self.n_topics
        random_seed = self.random_seed
        random_state = self.model.random_state

        # refit model with same random seed and verify results identical
        model_new = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_seed)
        rands_init = model_new._rands.copy()
        doc_topic_new = model_new.fit_transform(dtm)
        rands_fit = model_new._rands.copy()
        random_state_new = model_new.random_state
        np.testing.assert_array_equal(doc_topic_new, doc_topic)
        np.testing.assert_array_equal(random_state_new, random_state)

        # verify random variates are not changed
        np.testing.assert_array_equal(rands_init, rands_fit)

    def test_lda_monotone(self):
        dtm = self.dtm
        model = self.model
        n_topics = self.n_topics
        random_seed = self.random_seed

        # fit model with additional iterations, verify improvement in log likelihood
        n_iter = self.n_iter * 2
        model_new = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_seed)
        model_new.fit(dtm)
        self.assertGreater(model_new.loglikelihood(), model.loglikelihood())

    def test_lda_zero_iter(self):
        dtm = self.dtm
        model = self.model
        doc_topic = self.doc_topic
        n_topics = self.n_topics
        random_seed = self.random_seed

        # fit a new model with 0 iterations
        n_iter = 0
        model_new = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_seed)
        doc_topic_new = model_new.fit_transform(dtm)
        self.assertIsNotNone(model_new)
        self.assertIsNotNone(doc_topic_new)
        self.assertLess(model_new.loglikelihood(), model.loglikelihood())
        self.assertFalse((doc_topic_new == doc_topic).all())
