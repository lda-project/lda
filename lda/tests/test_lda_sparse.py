# coding=utf-8
from __future__ import absolute_import, unicode_literals  # noqa
import os


import numpy as np
import oslotest.base
import scipy.sparse

import lda
import lda.utils


class TestLDASparse(oslotest.base.BaseTestCase):

    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(__file__)
        reuters_ldac_fn = os.path.join(test_dir, 'reuters.ldac')
        cls.dtm = scipy.sparse.csr_matrix(lda.utils.ldac2dtm(open(reuters_ldac_fn), offset=0)).astype(np.int64)
        cls.n_iter = n_iter = 1
        cls.n_topics = n_topics = 10
        cls.random_seed = random_seed = 1
        cls.model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_seed)

    def test_lda_sparse(self):
        dtm = self.dtm
        model = self.model
        doc_topic = model.fit_transform(dtm)
        self.assertEqual(len(doc_topic), dtm.shape[0])
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

    def test_lda_sparse_error_float(self):
        dtm = self.dtm.astype(np.float)
        model = self.model
        self.assertRaises(ValueError, model.transform, dtm)
