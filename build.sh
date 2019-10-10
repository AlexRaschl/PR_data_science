#!/usr/bin/env bash

echo 'Deleting old build...'
rm -r build
echo ''

echo 'New build...'
python setup.py build
echo ''
