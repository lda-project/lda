from __future__ import absolute_import, unicode_literals  # noqa

import logging

from lda.lda import LDA  # noqa
import lda.datasets  # noqa

try:
    from importlib.metadata import version  # py38 and higher
    __version__ = version("lda")
except:  # noqa
    pass

logging.getLogger('lda').addHandler(logging.NullHandler())
