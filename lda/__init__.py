from __future__ import absolute_import, unicode_literals  # noqa

import logging

import pbr.version

from lda.lda import LDA  # noqa
import lda.datasets  # noqa

__version__ = pbr.version.VersionInfo('lda').version_string()

logging.getLogger('lda').addHandler(logging.NullHandler())
