import subprocess
import json
import os
from glob import glob
from dace import SDFG
# TODO: Change to LIT tests
folderPath = os.path.abspath(os.path.dirname(__file__))

def check_import(path):
    with open(path) as f:
        if 'XFAIL' in f.read():
            return 0

    result = subprocess.run([folderPath + '/../build/bin/sdir-translate', '--mlir-to-sdfg', path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    if(result.returncode != 0):
        return 1
    
    if(result.stdout.decode('utf-8') == ""):
        return 0
    
    translated = result.stdout.decode('utf-8')
    translated_json = json.loads(translated)

    try:
        sdfg = SDFG.from_json(translated_json)
        sdfg.save(filename=os.devnull, use_pickle=False, hash=None, exception=None)
        return 0
    except:
        return 1

anyFailed = False
for testPath in glob(folderPath + "/**/*.mlir", recursive=True):
    shortPath = testPath.replace(folderPath, "")
    #print("Testing: " + shortPath)
    if(check_import(testPath) != 0):
        print(shortPath + " failed!")
        anyFailed = True

if(anyFailed):
    exit(1)

print("All tests passed!")
exit(0)
