update-source: lda/_lda.c

lda/_lda.c: lda/_lda.pyx
	cython lda/_lda.pyx
