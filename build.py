import shutil
import subprocess
from pathlib import Path

BUILD_DIR = Path(__file__).parent.joinpath("build")
LIB_DIR = Path(__file__).parent.joinpath("lda")


def _meson(*args):
    """Invoke meson with the given arguments."""
    subprocess.check_call(["meson", *list(args)])


def _cleanup():
    """Remove build artifacts."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    for file in LIB_DIR.glob("*.so"):
        file.unlink()


def build():
    """Build the project."""
    _cleanup()

    _meson("setup", BUILD_DIR.as_posix())
    _meson("compile", "-C", BUILD_DIR.as_posix())
    _meson("install", "-C", BUILD_DIR.as_posix())


if __name__ == "__main__":
    build()
