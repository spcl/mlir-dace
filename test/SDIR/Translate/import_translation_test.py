import json
import sys
from dace import SDFG

SDFG.from_json(json.load(sys.stdin)).validate()
