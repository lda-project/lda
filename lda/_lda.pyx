#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

cimport cython
from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free


cdef int searchsorted(double* arr, int length, double value):
    """Cython version of numpy.searchsorted (bisection search)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz,
                   double[:] alpha, double[:] eta, double[:] rands):
    cdef int i, k, w, d, z, z_new
    cdef double r, dist_cum
    cdef int N = WS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]
    cdef double alpha_sum = 0
    for i in range(alpha.shape[0]):
        alpha_sum += alpha[i]
    cdef double eta_sum = 0
    for i in range(eta.shape[0]):
        eta_sum += eta[i]
    cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))

    for i in range(N):
        w = WS[i]
        d = DS[i]
        z = ZS[i]

        dec(nzw[z, w])
        dec(ndz[d, z])
        dec(nz[z])

        dist_cum = 0
        for k in range(n_topics):
            # eta is a double so cdivision yields a double
            dist_cum += (nzw[k, w] + eta[w]) / (nz[k] + eta_sum) * (ndz[d, k] + alpha[k])
            dist_sum[k] = dist_cum

        r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
        z_new = searchsorted(dist_sum, n_topics, r)

        ZS[i] = z_new
        inc(nzw[z_new, w])
        inc(ndz[d, z_new])
        inc(nz[z_new])

    free(dist_sum)
