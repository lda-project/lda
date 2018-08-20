PYTHON ?= python

cython:
	find lda -name "*.pyx" -exec $(PYTHON) -m cython -3 {} \;
