==============================
 How to make a release of lda
==============================

Note that ``pbr`` requires tags to be signed for its version calculations.

First build the source distribution with the following commands:

#. Run ``make cython`` so sdist can find the Cython-generated c files.
#. Tag (signed) the commit with the relevant version number
#. Fast-forward the ``master`` branch to ``develop``
#. Build source package with ``python setup.py sdist``

Now build the Windows and OS X wheels:

#. Build windows wheels, place them in ``dist/``
#. Build OS X wheels, place them in ``dist/``

Finally, upload everything to the Python Package index:

#. Upload and sign everything in ``dist/`` with ``twine upload --sign dist/*``
#. Upload documentation, ``python setup.py upload_docs``
