import os


def sdist_pre_hook(cmdobj):
    """Ensure Cython has compiled all pyx files to c."""
    print('sdist pre-hook')
    _pyxfiles = []
    for root, dirs, files in os.walk(__package__):
        for f in files:
            if f.endswith('.pyx'):
                _pyxfiles.append(os.path.join(root, f))
    for pyxfile in _pyxfiles:
        cfile = os.path.splitext(pyxfile)[0] + '.c'
        msg = ("C source file '{}' not found. ".format(cfile) +
               "Run 'make cython' before sdist.")
        assert os.path.isfile(cfile), msg
