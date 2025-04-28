import subprocess
import time
import sys
import os

python_executable = sys.executable

this_folder = os.path.dirname(__file__)
script_path = os.path.join(this_folder, "preprocessdatavideos.py")

while True:
    print("Starting preprocessdatavideos.py ...")
    subprocess.run([python_executable, script_path])
    print("preprocessdatavideos.py done. Restarting...")
    time.sleep(1)
