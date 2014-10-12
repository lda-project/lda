==============================
 How to make a release of lda
==============================

Note that ``pbr`` requires tags to be signed for its version calculations.

#. Run ``make update-source`` and commit the changes
#. Tag (signed) the commit with the relevant version number
#. Build source package with ``python setup.py sdist``
#. Build windows wheels, place them in ``dist/``
#. Build OS X wheels, place them in ``dist/``
#. Upload everything in ``dist/`` with ``twine upload --sign dist/*``
#. Upload documentation, ``python setup.py upload_docs``
