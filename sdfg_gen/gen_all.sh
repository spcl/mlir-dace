#!/bin/bash
cd "$(dirname "$0")"
rm -rf gen/json/*
rm -rf gen/sdfg/*
for f in sdfg/*.py; do python "$f"; done