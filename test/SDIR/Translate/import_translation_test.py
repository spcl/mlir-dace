import json
import os
from dace import SDFG


def check_import(string):
    try:
        translated_json = json.loads(string)
        sdfg = SDFG.from_json(translated_json)
        sdfg.validate()
        sdfg.save(filename=os.devnull)
        return 0
    except:
        return 1


check_import(os.sys.stdin)
