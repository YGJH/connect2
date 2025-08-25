import subprocess
import time
while True:
    subprocess.run(['uv', 'run' , 'remove_garbege_model.py'], check=True)
    time.sleep(60)

