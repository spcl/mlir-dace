#!/bin/bash
cd "$(dirname "$0")"
rm -r gen
for f in sdfg/*.py; do python "$f"; done