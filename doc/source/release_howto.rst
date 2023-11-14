==============================
 How to make a release of lda
==============================

Fingerprint of signing key is ``94D5E5A35ED429648B1C627AD96242D5314C8249``.

1. Verify that the following chores are finished:

    - Tests pass.
    - Changes since last release are mentioned in ``doc/source/whats_new.rst``.
    - Signed tag for the current release exists. Run ``git tag -s -u 94D5E5A35ED429648B1C627AD96242D5314C8249 <n.n.n>``.
    - The version in ``pyproject.toml`` is the same as in the tag.

2. Push the tag to GitHub; a GitHub Action will automatically publish the wheels and source dist to PyPI.
