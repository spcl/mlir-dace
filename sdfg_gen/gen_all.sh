#!/bin/bash

cd "$(dirname "$0")" || exit
rm -rf gen/json/*
rm -rf gen/sdfg/*
for f in python/*.py; do python3 "$f"; done
