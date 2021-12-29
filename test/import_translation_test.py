import subprocess
import json
import os
from glob import glob
from dace import SDFG

# TODO: Add this test to CMake tests


def check_import(path):
    with open(path) as f:
        if 'XFAIL' in f.read():
            return 0

    result = subprocess.run(
        [folderPath + '/../build/bin/sdir-translate', '--mlir-to-sdfg', path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)

    if (result.returncode != 0):
        return 1

    translated = result.stdout.decode('utf-8')

    if (translated == ""):
        return 0

    try:
        translated_json = json.loads(translated)
        sdfg = SDFG.from_json(translated_json)
        sdfg.save(filename=os.devnull)
        return 0
    except:
        return 1


folderPath = os.path.abspath(os.path.dirname(__file__))
anyFailed = False

for testPath in glob(folderPath + "/**/*.mlir", recursive=True):
    shortPath = testPath.replace(folderPath, "")
    # To manually check all the test files
    #print("Testing: " + shortPath)
    if (check_import(testPath) != 0):
        print(shortPath + " failed!")
        anyFailed = True

if (anyFailed):
    exit(1)

print("All tests passed!")
exit(0)
