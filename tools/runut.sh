#!/bin/bash

# Run a specified unit test

export PYTHONPATH=$(pwd)/python:$PYTHONPATH

cd $(dirname $0)/..
if [ $# -eq 1 ]; then
  make setuputs || exit 1
  make libfemtools || exit 1
fi
cd $(dirname $1)
perl -pi -e 's/$1//' Makefile
rm $(basename $1)
make $(basename $1) || exit 1
cd ../../bin/tests
ln -s ../../$1
exec ./$(basename $1)
