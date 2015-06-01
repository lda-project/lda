from __future__ import absolute_import, unicode_literals  # noqa

import logging
import numbers
import sys

import numpy as np

PY2 = sys.version_info[0] == 2
if PY2:
    import itertools
    zip = itertools.izip


logger = logging.getLogger('lda')


def check_random_state(seed):
    if seed is None:
        # i.e., use existing RandomState
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("{} cannot be used as a random seed.".format(seed))


def matrix_to_lists(doc_word):
    """Convert a (sparse) matrix of counts into arrays of word and doc indices

    Parameters
    ----------
    doc_word : array or sparse matrix (D, V)
        document-term matrix of counts

    Returns
    -------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    """
    if np.count_nonzero(doc_word.sum(axis=1)) != doc_word.shape[0]:
        logger.warning("all zero row in document-term matrix found")
    if np.count_nonzero(doc_word.sum(axis=0)) != doc_word.shape[1]:
        logger.warning("all zero column in document-term matrix found")
    sparse = True
    try:
        # if doc_word is a scipy sparse matrix
        doc_word = doc_word.copy().tolil()
    except AttributeError:
        sparse = False

    if sparse and not np.issubdtype(doc_word.dtype, int):
        raise ValueError("expected sparse matrix with integer values, found float values")

    ii, jj = np.nonzero(doc_word)
    if sparse:
        ss = tuple(doc_word[i, j] for i, j in zip(ii, jj))
    else:
        ss = doc_word[ii, jj]

    n_tokens = int(doc_word.sum())
    DS = np.repeat(ii, ss).astype(np.intc)
    WS = np.empty(n_tokens, dtype=np.intc)
    startidx = 0
    for i, cnt in enumerate(ss):
        cnt = int(cnt)
        WS[startidx:startidx + cnt] = jj[i]
        startidx += cnt
    return WS, DS


def lists_to_matrix(WS, DS):
    """Convert array of word (or topic) and document indices to doc-term array

    Parameters
    -----------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    Returns
    -------
    doc_word : array (D, V)
        document-term array of counts

    """
    D = max(DS) + 1
    V = max(WS) + 1
    doc_word = np.empty((D, V), dtype=np.intc)
    for d in range(D):
        for v in range(V):
            doc_word[d, v] = np.count_nonzero(WS[DS == d] == v)
    return doc_word


def dtm2ldac(dtm, offset=0):
    """Convert a document-term matrix into an LDA-C formatted file

    Parameters
    ----------
    dtm : array of shape N,V

    Returns
    -------
    doclines : iterable of LDA-C lines suitable for writing to file

    Notes
    -----
    If a format similar to SVMLight is desired, `offset` of 1 may be used.
    """
    try:
        dtm = dtm.tocsr()
    except AttributeError:
        pass
    assert np.issubdtype(dtm.dtype, int)
    n_rows = dtm.shape[0]
    for i, row in enumerate(dtm):
        try:
            row = row.toarray().squeeze()
        except AttributeError:
            pass
        unique_terms = np.count_nonzero(row)
        if unique_terms == 0:
            raise ValueError("dtm row {} has all zero entries.".format(i))
        term_cnt_pairs = [(i + offset, cnt) for i, cnt in enumerate(row) if cnt > 0]
        docline = str(unique_terms) + ' '
        docline += ' '.join(["{}:{}".format(i, cnt) for i, cnt in term_cnt_pairs])
        if (i + 1) % 1000 == 0:
            logger.info("dtm2ldac: on row {} of {}".format(i + 1, n_rows))
        yield docline


def ldac2dtm(stream, offset=0):
    """Convert an LDA-C formatted file to a document-term array

    Parameters
    ----------
    stream: file object
        File yielding unicode strings in LDA-C format.

    Returns
    -------
    dtm : array of shape N,V

    Notes
    -----
    If a format similar to SVMLight is the source, an `offset` of 1 may be used.
    """
    doclines = stream

    # We need to figure out the dimensions of the dtm.
    N = 0
    V = -1
    data = []
    for l in doclines:
        l = l.strip()
        # skip empty lines
        if not l:
            continue
        unique_terms = int(l.split(' ')[0])
        term_cnt_pairs = [s.split(':') for s in l.split(' ')[1:]]
        for v, _ in term_cnt_pairs:
            # check that format is indeed LDA-C with the appropriate offset
            if int(v) == 0 and offset == 1:
                raise ValueError("Indexes in LDA-C are offset 1")
        term_cnt_pairs = tuple((int(v) - offset, int(cnt)) for v, cnt in term_cnt_pairs)
        np.testing.assert_equal(unique_terms, len(term_cnt_pairs))
        V = max(V, *[v for v, cnt in term_cnt_pairs])
        data.append(term_cnt_pairs)
        N += 1
    V = V + 1
    dtm = np.zeros((N, V), dtype=np.intc)
    for i, doc in enumerate(data):
        for v, cnt in doc:
            np.testing.assert_equal(dtm[i, v], 0)
            dtm[i, v] = cnt
    return dtm
