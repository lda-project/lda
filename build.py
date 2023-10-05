from Cython.Distutils import build_ext
from setuptools import Extension


def build(kwargs: dict):
    """Build and cythonize the extension.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments passed to the setup function.
    """
    kwargs["ext_modules"] = [
        Extension("lda._lda", sources=["src/lda/_lda.pyx", "src/lda/gamma.c"])
    ]
    kwargs["cmdclass"] = {"build_ext": build_ext}
