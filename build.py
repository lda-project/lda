from Cython.Build import cythonize
from setuptools import Extension


def build(kwargs: dict):
    """Build and cythonize the extension.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments passed to the setup function.
    """
    extension = Extension("lda._lda", sources=["lda/_lda.pyx", "lda/gamma.c"])
    kwargs["ext_modules"] = cythonize([extension])
