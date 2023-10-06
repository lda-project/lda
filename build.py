from Cython.Build import cythonize
from setuptools import Extension


def build(kwargs: dict):
    """Cythonize and build the extension.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments passed to the setup function.
    """
    kwargs["ext_modules"] = cythonize(
        [
            Extension(
                "lda._lda",
                sources=["lda/_lda.pyx", "lda/gamma.c"],
                include_dirs=["lda"],
            )
        ]
    )
