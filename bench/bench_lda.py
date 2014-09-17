#!/usr/bin/python3

import os
import time

import lda
import lda.utils

test_dir = os.path.join(os.path.dirname(__file__), '..', 'lda', 'tests')
reuters_ldac_fn = os.path.join(test_dir, 'reuters.ldac')
dtm = lda.utils.ldac2dtm(open(reuters_ldac_fn), offset=0)
t0 = time.time()
n_iter = 100
n_topics = 100
random_seed = 1
model = model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_seed)

t0 = time.time()
doc_topic = model.fit_transform(dtm)
print("seconds elapsed: {}".format(time.time() - t0))
