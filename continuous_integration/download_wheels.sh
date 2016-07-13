#!/bin/bash
# script to download lda wheels for release
RACKSPACE_URL=https://d36102825770f036e7f0-25e1da3ee193e97cce4726db07962f5d.ssl.cf5.rackcdn.com/
if [ "`which twine`" == "" ]; then
    echo "twine not on path; need to pip install twine?"
    exit 1
fi
LDA_VERSION=`git describe --tags`
WHEEL_HEAD="lda-${LDA_VERSION}"
WHEEL_TAIL32="win32.whl"
WHEEL_TAIL64="win_amd64.whl"
mkdir -p wheels
cd wheels
rm -rf *.whl
for py_tag in cp27-none cp33-none cp34-none
do
    wheel_name="$WHEEL_HEAD-$py_tag-$WHEEL_TAIL32"
    wheel_url="${RACKSPACE_URL}/${wheel_name}"
    curl -O $wheel_url
    wheel_name="$WHEEL_HEAD-$py_tag-$WHEEL_TAIL64"
    wheel_url="${RACKSPACE_URL}/${wheel_name}"
    curl -O $wheel_url
done
# suggest upload to pypi
cd ..
echo "wheels now need to be uploaded to PyPI using a command such as:"
echo "twine upload --sign wheels/*.whl"
