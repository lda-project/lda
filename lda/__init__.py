from __future__ import absolute_import, unicode_literals  # noqa

import logging

import pbr.version

from lda.lda import LDA  # noqa

__version__ = pbr.version.VersionInfo('lda').version_string()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('lda')
logger.addHandler(logging.NullHandler())
