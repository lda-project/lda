================
Benchmarking lda
================

This directory contains scripts to compare the running time of ``lda`` against
hca_. hca_ is written entirely in C.

To run ``bench_hca`` you will need to have hca_ on your path.

The test uses the following settings for hca_

- 100 topics
- 100 iterations
- Latent Dirichlet allocation (used automatically with ``-A<float>`` and ``-B<float>``)

.. _hca: http://www.mloss.org/software/view/527/
