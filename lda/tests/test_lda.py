# coding=utf-8
from __future__ import absolute_import, unicode_literals  # noqa

import numpy as np
import oslotest.base

import lda


class TestLDA(oslotest.base.BaseTestCase):

    def test_lda_constructor(self):
        n_topics = 10
        model1 = lda.LDA(n_topics)
        self.assertIsNotNone(model1)
        model2 = lda.LDA(n_topics=n_topics)
        self.assertIsNotNone(model2)

    def test_lda_params(self):
        n_topics = 10
        model1 = lda.LDA(n_topics, alpha=0.3)
        self.assertIsNotNone(model1)
        model2 = lda.LDA(n_topics=n_topics, alpha=0.3, eta=0.4)
        self.assertIsNotNone(model2)
        self.assertRaises(ValueError, lda.LDA, n_topics, alpha=-3)
        self.assertRaises(ValueError, lda.LDA, n_topics, eta=-3)
        self.assertRaises(ValueError, lda.LDA, n_topics, alpha=-3, eta=-3)

    def test_lda_getting_started(self):
        X = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
        model = lda.LDA(n_topics=2, n_iter=100, random_state=1)
        doc_topic = model.fit_transform(X)
        self.assertIsNotNone(doc_topic)
        self.assertIsNotNone(model.doc_topic_)
        self.assertIsNotNone(model.components_)

    def test_lda_loglikelihoods(self):
        X = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
        model = lda.LDA(n_topics=2, n_iter=100, random_state=1)
        model.fit(X)
        self.assertGreater(len(model.loglikelihoods_), 1)
