import subprocess
import json
import os
from glob import glob
from dace import SDFG

# TODO: Add this test to CMake tests

folderPath = os.path.abspath(os.path.dirname(__file__))


def check_import(path):
    xfail = False

    with open(path) as f:
        if 'XFAIL' in f.read():
            xfail = True

    result = subprocess.run(
        [folderPath + '/../build/bin/sdir-translate', '--mlir-to-sdfg', path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)

    if (result.returncode != xfail):
        return 1

    translated = result.stdout.decode('utf-8')

    if (translated == ""):
        return 0

    try:
        translated_json = json.loads(translated)
        sdfg = SDFG.from_json(translated_json)
        sdfg.validate()
        sdfg.save(filename=os.devnull)
        return 0
    except:
        return 1


anyFailed = False

for testPath in glob(folderPath + "/**/*.mlir", recursive=True):
    shortPath = "test" + testPath.replace(folderPath, "")

    os.sys.stdout = open(os.devnull, 'w')
    os.sys.stderr = open(os.devnull, 'w')
    testResult = check_import(testPath)
    os.sys.stdout = os.sys.__stdout__
    os.sys.stderr = os.sys.__stderr__

    if (testResult != 0):
        print("\u274c " + shortPath)
        anyFailed = True
    else:
        print("\u2705 " + shortPath)

if (anyFailed):
    exit(1)

print("All tests passed!")
exit(0)
