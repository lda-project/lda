PYTHON ?= python
CYTHON ?= cython

cython:
	find lda -name "*.pyx" -exec $(CYTHON) {} \;
