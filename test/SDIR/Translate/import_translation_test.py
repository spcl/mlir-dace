import json
import sys
from dace import SDFG

try:
    SDFG.from_json(json.load(sys.stdin)).validate()
except Exception as e:
    print(e)
    exit(1)
