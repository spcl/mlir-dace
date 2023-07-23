# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

# Executes given SDFG with three-filled arrays and prints all output to stdout

import json
import sys
import dace
from dace import SDFG
from dace.config import Config

Config.set("cache", value='unique')

sdfg = SDFG.from_json(json.load(sys.stdin))
obj = sdfg.compile()

arg_dict = {}

for arg_name, arg_type in sdfg.arglist().items():
    array = dace.ndarray(shape=arg_type.shape, dtype=arg_type.dtype)
    array.fill(3)
    arg_dict[arg_name] = array

obj(**arg_dict)
latest_results = arg_dict

for arg_name, array in latest_results.items():
    print("begin_dump: %s" % arg_name)
    for elem in array:
        print(elem)
    print("end_dump: %s" % arg_name)
