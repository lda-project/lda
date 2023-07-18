# coding=utf-8
from __future__ import absolute_import, unicode_literals  # noqa

import oslotest.base

import lda.datasets


class TestDatasets(oslotest.base.BaseTestCase):

    def test_datasets(self):
        X = lda.datasets.load_reuters()
        self.assertEqual(X.shape, (395, 4258))
        titles = lda.datasets.load_reuters_titles()
        self.assertEqual(len(titles), X.shape[0])
        vocab = lda.datasets.load_reuters_vocab()
        self.assertEqual(len(vocab), X.shape[1])
