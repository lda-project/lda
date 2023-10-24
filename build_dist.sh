#!/bin/bash
# bash strict mode
set -euo pipefail
IFS=$'\n\t'

err() {
  echo "$@" >&2
}

LAST=$(git tag --sort version:refname | grep -v rc | tail -1)

echo "Building distribution for: $LAST"
git checkout $LAST

# superficial check that whats_new.rst has been updated
if ! grep -q "$LAST" doc/source/whats_new.rst ; then
  err "release notes not updated, exiting"
  exit -1
fi

read -p "Ok to continue (y/n)? " answer
case ${answer:0:1} in
    y|Y )
        echo "Building distribution"
        poetry build
    ;;
    * )
        echo "Not building distribution"
    ;;
esac
